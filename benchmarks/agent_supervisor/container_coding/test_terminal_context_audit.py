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


def test_generic_retrieval_scope_must_match_instruction_bound_signed_profile(compiled):
    from benchmarks.agent_supervisor.container_coding.terminal_task_profile import PROFILE, instruction_sha256, task_profile_bytes
    path = compiled["state"] / "prepared.json"
    prepared = json.loads(path.read_text())
    profile = {"schema": "terminal-public-task-profile@1",
        "instruction_sha256": instruction_sha256(prepared["query"]),
        "input_paths": ["bottle.py"],
        "outputs": [{"path": "bottle.py", "effect": "modify", "media_type": "text/x-python"}]}
    prepared["task_profile"] = profile
    prepared["manifest"]["payload"]["sources"][PROFILE] = {
        "sha256": hashlib.sha256(task_profile_bytes(profile)).hexdigest()}
    path.write_text(json.dumps(prepared))
    assert observe(compiled)["all_observed_coding_inputs_verified"] is True
    prepared["task_profile"]["instruction_sha256"] = "0" * 64
    path.write_text(json.dumps(prepared))
    assert observe(compiled)["all_observed_coding_inputs_verified"] is False


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


def _rewrite_formalization(fixture, payload, *, world_snapshot_cid=None):
    """Compile genuine native references and update the matching router digests."""
    from ipfs_accelerate_py.agent_supervisor.context.context_contracts import (
        ContextCapsule, canonical_context_json_bytes,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _observation
    original = ContextCapsule.from_dict(json.loads(fixture["capsule"].read_text()))
    refs = [row for row in original.evidence if row.kind != "source-formalization-context"]
    if world_snapshot_cid is not None:
        world = json.loads("".join(row.summary for row in refs if row.kind == "intent-world-context"))
        world["world_snapshot_cid"] = world_snapshot_cid
        refs = [row for row in refs if row.kind != "intent-world-context"]
        refs.extend(build_text_context_references(json.dumps(world, ensure_ascii=False),
            reference_prefix="context-2", kind="intent-world-context", chunk_bytes=613, required=True))
    text = canonical_context_json_bytes(payload).decode() if payload is not None else None
    if text is not None:
        refs.extend(build_text_context_references(text, reference_prefix="source-formalization",
            kind="source-formalization-context", required=True, chunk_bytes=6144,
            path=payload["context_bundle"]["artifact"], repository_id="repo:fixture", tree_id="tree:fixture"))
    capsule = ContextCompiler(ContextBudget(max_input_tokens=20000, max_items=64,
        max_item_bytes=16384, max_serialized_bytes=262144)).compile(
        repository_id="repo:fixture", tree_id="tree:fixture", objective_id="TB-CODE-TASK",
        objective_revision="sha256:task", policy_id="policy:fixture", policy_revision="sha256:policy",
        caller="supervisor:fixture", stage="implementation", goal={"id": "TB-CODE-TASK"},
        authority={"mode": "proposal"}, scope={"paths": ["bottle.py"]},
        acceptance={"criteria": ["pending public check"]}, evidence=refs).capsule
    fixture["capsule"].write_text(capsule.to_json())
    native = render_context_capsule(capsule)
    receipt = fixture["receipts"][0]
    model, header = render_model_prompt(prompt=native, purpose="coding", workspace=Path(receipt["workspace"]))
    receipt.update(prompt_sha256=sha(native), prompt_bytes=len(native.encode()),
        native_prompt_sha256=sha(native), native_prompt_bytes=len(native.encode()),
        model_prompt_sha256=sha(model), model_prompt_bytes=len(model.encode()),
        workspace_advisory_sha256=sha(header), workspace_advisory_bytes=len(header.encode()))
    if text is not None:
        receipt["source_formalization_context"] = _observation(payload, text,
            sum(row.kind == "source-formalization-context" for row in refs), {"bottle.py": "captured"})
    else:
        receipt.pop("source_formalization_context", None)


@pytest.fixture
def formalization(compiled, tmp_path, monkeypatch):
    """Authored transport receipts; native bundles/compilation, no model claim."""
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundle
    from ipfs_accelerate_py.agent_supervisor.runtime import source384_repository_context as source384
    from ipfs_accelerate_py.agent_supervisor.runtime import numerical_formalization_context as numerical
    root = tmp_path / "canonical-repository"
    root.mkdir()
    (root / "bottle.py").write_text("original bottle source")
    prepared_path, expected_path = (compiled["state"] / name for name in ("prepared.json", "context-result.json"))
    prepared = json.loads(prepared_path.read_text())
    prepared["repository"] = str(root)
    prepared_path.write_text(json.dumps(prepared))
    hashes = {"bottle.py": prepared["manifest"]["payload"]["sources"]["bottle.py"]["sha256"]}
    selected = {}
    for key, schema in (("source384_context", source384.SCHEMA), ("numerical_formalization", numerical.SCHEMA)):
        output = tmp_path / key
        output.mkdir()
        receipt = dict(schema=schema, repository=str(root), output=str(output), source_hashes=hashes,
            summary={"schema": "authored-audit-summary@1", "private_candidate": "private candidate λ" * 100,
                "completion_authority": False}, completion_authority=False)
        raw = bundle._raw(receipt)
        (output / "receipt.json").write_bytes(raw)
        if key == "numerical_formalization":
            receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
        selected[key] = receipt
    nomination = dict(schema="supervisor-task-context-preparation@1",
        task_cid="cid:task", task_id="TB-CODE-TASK", metadata={}, **selected)
    binding = bundle.write_task_context_bundle(repository=root, prepared=[nomination], output=root / "bundle.json")
    expected = json.loads(expected_path.read_text())
    expected.update(context_bundle=binding, **selected)
    expected_path.write_text(json.dumps(expected))
    payload = bundle._worker_evidence(selected, repository=root, artifact=binding["artifact"],
        expected_sha256=binding["sha256"], task_cid="cid:task", task_id="TB-CODE-TASK")
    compiled.update(repository=root, selected=selected, nomination=nomination, evidence=payload)
    _rewrite_formalization(compiled, payload)

    def forbidden(**kwargs):
        pytest.fail("historical audit must not replay a live owner or inference")
    monkeypatch.setattr(source384, "validate_source384_context", forbidden)
    monkeypatch.setattr(numerical, "validate_numerical_context", forbidden)
    monkeypatch.setattr(bundle, "load_task_context_selection", forbidden)
    return compiled


@pytest.mark.parametrize("inherited", [False, True])
def test_historical_formalization_joins_survive_edits_without_authority(formalization, inherited):
    receipt = formalization["receipts"][0]["source_formalization_context"]
    receipt.update(owner_validation_inherited=inherited, owner_private_state_replayed_here=not inherited)
    (formalization["repository"] / "bottle.py").write_text("worker completed an intentional source edit")
    result = observe(formalization)
    assert result["all_observed_coding_inputs_verified"] is True, result
    assert result["matches"][0]["model_input_checks"]["source_formalization_dispatch"] is True
    observation = result["capsules"][0]["source_formalization_context"]
    assert observation["historical_receipt_bindings_verified"] is True
    assert observation["current_sources_checked"] is observation["neural_inference_replayed"] is False
    assert observation["source_obligations_discharged"] == 0
    assert result["proof_authority"] is result["formalization_authority"] is False
    assert len(observation["contexts"]) == 2
    assert not formalization["workspace_root"].exists()
    exported = json.dumps(result)
    assert "private candidate" not in exported and formalization["query"] not in exported
    assert formalization["receipts"][0]["workspace"] not in exported


def test_bounded_child_preserves_formalization_dispatch_observation(formalization):
    (formalization["repository"] / "bottle.py").unlink()
    result = audit.collect_terminal_context_audit(state=formalization["state"],
        receipts=formalization["receipts"], workspace_root=formalization["workspace_root"], timeout_seconds=10)
    assert result["all_observed_coding_inputs_verified"] is True, result
    assert result["matches"][0]["model_input_checks"]["source_formalization_dispatch"] is True


def test_independent_selection_detects_omitted_prompt_and_matching_router_receipt(formalization):
    _rewrite_formalization(formalization, None)
    result = observe(formalization)
    assert result["all_observed_coding_inputs_verified"] is False
    assert not result["matches"] and result["read_errors"]


def test_prompt_cannot_self_nominate_unselected_formalization(formalization):
    expected_path = formalization["state"] / "context-result.json"
    expected = json.loads(expected_path.read_text())
    for key in ("source384_context", "numerical_formalization", "context_bundle"):
        expected.pop(key)
    expected_path.write_text(json.dumps(expected))
    assert observe(formalization)["all_observed_coding_inputs_verified"] is False


def test_valid_dispatch_observation_does_not_introduce_unselected_advice(formalization):
    observation = formalization["receipts"][0]["source_formalization_context"]
    _rewrite_formalization(formalization, None)
    formalization["receipts"][0]["source_formalization_context"] = observation
    path = formalization["state"] / "context-result.json"
    expected = json.loads(path.read_text())
    for key in ("source384_context", "numerical_formalization", "context_bundle"):
        expected.pop(key)
    path.write_text(json.dumps(expected))
    result = observe(formalization)
    assert result["any_native_input_verified"] is True
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["matches"][0]["model_input_checks"]["source_formalization_dispatch"] is False


def test_unmapped_warm_world_cannot_be_admitted_by_exact_formalization_receipts(formalization):
    # The main benchmark nominates the original capture. A future warm-preflight
    # final audit must independently nominate its newly captured world; matching
    # formalization receipt bytes cannot confer that world-selection authority.
    _rewrite_formalization(formalization, formalization["evidence"], world_snapshot_cid="cid:warm-world")
    result = observe(formalization)
    assert result["all_observed_coding_inputs_verified"] is False
    assert result["capsules"][0]["checks"]["world_root"] is False


def test_rehashed_source_claims_still_require_independently_captured_manifest(formalization):
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundle
    root = formalization["repository"]
    selected = copy.deepcopy(formalization["selected"])
    for key, receipt in selected.items():
        receipt["source_hashes"]["bottle.py"] = "f" * 64
        raw = bundle._raw({k: v for k, v in receipt.items() if k != "receipt_sha256"})
        (Path(receipt["output"]) / "receipt.json").write_bytes(raw)
        if key == "numerical_formalization":
            receipt["receipt_sha256"] = hashlib.sha256(raw).hexdigest()
    binding = bundle.write_task_context_bundle(repository=root,
        prepared=[{**formalization["nomination"], **selected}], output=root / "rehashed.json")
    path = formalization["state"] / "context-result.json"
    expected = json.loads(path.read_text())
    expected.update(context_bundle=binding, **selected)
    path.write_text(json.dumps(expected))
    payload = bundle._worker_evidence(selected, repository=root, artifact=binding["artifact"],
        expected_sha256=binding["sha256"], task_cid="cid:task", task_id="TB-CODE-TASK")
    _rewrite_formalization(formalization, payload)
    result = observe(formalization)
    assert result["all_observed_coding_inputs_verified"] is False
    assert not result["matches"] and result["read_errors"]


@pytest.mark.parametrize("change", ["missing", "null", "task", "bundle", "evidence", "source_count",
    "authority", "authority_zero", "discharged_false", "inherited_integer", "extra", "source_kind"])
def test_formalization_dispatch_contract_cannot_be_missing_or_forged(formalization, change):
    receipt = formalization["receipts"][0]
    recorded = receipt["source_formalization_context"]
    if change == "missing":
        receipt.pop("source_formalization_context")
    elif change == "null":
        receipt["source_formalization_context"] = None
    elif change == "task":
        recorded["task_cid"] = "cid:foreign-task"
    elif change in ("bundle", "evidence"):
        recorded["context_bundle_sha256" if change == "bundle" else "worker_evidence_sha256"] = "f" * 64
    elif change == "source_count":
        recorded["source_files"] += 1
    elif change.startswith("authority"):
        recorded["proof_authority"] = True if change == "authority" else 0
    elif change == "discharged_false":
        recorded["source_obligations_discharged"] = False
    elif change == "inherited_integer":
        recorded["owner_validation_inherited"] = 0
    elif change == "source_kind":
        recorded["context_kinds"].reverse()
    else:
        recorded["candidate_only"] = True
    result = observe(formalization)
    assert result["all_observed_coding_inputs_verified"] is not True
    assert not any(match["model_input_verified"] for match in result["matches"])


@pytest.mark.parametrize("change", ["source_receipt", "numerical_receipt", "bundle", "mirror_omission", "foreign_source"])
def test_historical_receipt_bundle_and_prepared_population_joins(formalization, change):
    if change in ("source_receipt", "numerical_receipt"):
        key = "source384_context" if change == "source_receipt" else "numerical_formalization"
        path = Path(formalization["selected"][key]["output"]) / "receipt.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "bundle":
        path = formalization["repository"] / "bundle.json"
        path.write_bytes(path.read_bytes() + b" ")
    elif change == "mirror_omission":
        path = formalization["state"] / "context-result.json"
        expected = json.loads(path.read_text())
        expected.pop("numerical_formalization")
        path.write_text(json.dumps(expected))
    else:
        payload = copy.deepcopy(formalization["evidence"])
        payload["contexts"][0]["source_hashes_sha256"] = "f" * 64
        _rewrite_formalization(formalization, payload)
    result = observe(formalization)
    assert result["all_observed_coding_inputs_verified"] is False
    assert not result["matches"] and result["read_errors"]


def test_task_identical_bundle_renomination_preserves_exact_original_receipts(formalization):
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundle
    root = formalization["repository"]
    rebound = bundle.write_task_context_bundle(repository=root,
        prepared=[{**formalization["nomination"], "metadata": {"world context artifact": "new-world.json",
            "world context sha256": "a" * 64, "world context repository": "repo:fixture"}}],
        output=root / "rebound.json")
    payload = bundle._worker_evidence(formalization["selected"], repository=root,
        artifact=rebound["artifact"], expected_sha256=rebound["sha256"], task_cid="cid:task", task_id="TB-CODE-TASK")
    _rewrite_formalization(formalization, payload)
    result = observe(formalization)
    assert result["all_observed_coding_inputs_verified"] is True, result
    observed = result["capsules"][0]["source_formalization_context"]
    assert observed["expected_context_bundle_sha256"] != observed["dispatched_context_bundle_sha256"]


@pytest.mark.parametrize("schema", ["supervisor-task-context-nominations@2", "supervisor-task-context-nominations@3"])
def test_legacy_source384_bundle_can_be_historically_joined(formalization, schema):
    from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundle
    root = formalization["repository"]
    source = formalization["selected"]["source384_context"]
    binding = bundle.write_task_context_bundle(repository=root,
        prepared=[{k: v for k, v in formalization["nomination"].items() if k != "numerical_formalization"}],
        output=root / "legacy.json")
    if schema.endswith("@2"):
        value = json.loads((root / "legacy.json").read_bytes())
        value["schema"] = schema
        value["tasks"][0]["source384_context"] = source
        raw = bundle._raw(value)
        (root / "legacy.json").write_bytes(raw)
        binding["sha256"] = hashlib.sha256(raw).hexdigest()
    path = formalization["state"] / "context-result.json"
    expected = json.loads(path.read_text())
    expected.pop("numerical_formalization")
    expected["context_bundle"] = binding
    path.write_text(json.dumps(expected))
    payload = bundle._worker_evidence({"source384_context": source}, repository=root,
        artifact=binding["artifact"], expected_sha256=binding["sha256"], task_cid="cid:task", task_id="TB-CODE-TASK")
    _rewrite_formalization(formalization, payload)
    assert observe(formalization)["all_observed_coding_inputs_verified"] is True
