"""Real signed full inventories with bounded selected context and native reuse.

All planner graphs are authored. Git, owner signatures, native vector/semantic
producers, Intent/DuckDB and the native-owner warm capture are real. No model,
training, Docker, target execution or official verifier is dispatched here.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_initial_context as initial
from benchmarks.agent_supervisor.container_coding.terminal_native_preflight import _authored_graph
from benchmarks.agent_supervisor.container_coding.terminal_context_rebind import rebind_full_context, verify_worker_context_prompt
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import git
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_semantic_scope import (
    replay_terminal_semantic_scope, terminal_semantic_scope_observation,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalTask, TodoImplementationDaemon

TEXT = "Inspect the selected functions and correct module_000.py.\n"


def make_case(tmp_path, *, byte_delta=None):
    root = tmp_path / "repository";root.mkdir()
    names = [f"module_{index:03d}.py" for index in range(70)]
    selected = names[:2] if byte_delta is None else names[:4]
    declared = dict(schema=profiles.SCOPED_SCHEMA,
        instruction_sha256=profiles.instruction_sha256(TEXT), input_paths=names,
        outputs=[dict(path=names[0], effect="modify", media_type="text/x-python")],
        semantic_scope=dict(schema=profiles.SEMANTIC_SCOPE_SCHEMA, input_paths=selected,
            required_input_paths=names[:1], program_paths=selected))
    source = {name: f"def value_{index:03d}():\n    return {index}\n".encode()
              for index, name in enumerate(names)}
    if byte_delta is not None:
        support_bytes = len(TEXT.encode()) + len(profiles.task_profile_bytes(declared)) + len(profiles.task_profile_smoke(declared).encode())
        remaining = 1_000_000 - support_bytes + byte_delta
        lengths = [remaining // 4 + (index < remaining % 4) for index in range(4)]
        for name, size in zip(selected, lengths):
            source[name] += b"#" + b"x" * (size-len(source[name])-2) + b"\n"
            assert len(source[name]) == size and size <= 262144
    git(root, "init", "-q")
    for name, raw in source.items(): (root/name).write_bytes(raw)
    git(root, "add", "--", *names)
    git(root, "-c", "user.name=Signed scope fixture", "-c", "user.email=scope@example.invalid", "commit", "-qm", "Authored inert originals")
    instruction = tmp_path / "instruction.md";instruction.write_text(TEXT)
    state = tmp_path / "state"
    prepared = prep.prepare(repository=root, instruction=instruction, state=state,
        task_profile=declared, disable_intent_autoencoder=True)
    return SimpleNamespace(root=root, state=state, prepared=prepared, declared=declared,
        source=source, selected=selected)


@pytest.fixture
def case(tmp_path):
    return make_case(tmp_path)


def admit(case):
    graph = _authored_graph(case.prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=case.prepared["manifest"])
    prep._write(case.state / "admission.json", admission)
    with IntentRepository(case.state / "intent.duckdb") as intent:
        local.materialize_local_benchmark_plan(admission=admission, intent=intent)
    return graph, admission


@pytest.mark.parametrize("ranked", [False, True])
def test_large_complete_inventory_builds_cold_loads_and_reuses_exact_selected_indexes(case, ranked):
    mode = "verified-retrieval-symbol-rank@1"
    receipt = prep.initial_context(state=case.state, **({"capsule_selection":mode} if ranked else {}))
    loaded = initial.load_initial_context(state=case.state, prepared=case.prepared, require_empty_owner=True)
    scope = replay_terminal_semantic_scope(case.prepared)
    assert len(case.prepared["worker_inputs"]) == len(case.prepared["manifest"]["payload"]["sources"]) == 73
    assert scope["full_count"] == 73 and scope["selected_count"] == 5 and scope["frontier_count"] == 68
    assert loaded["descriptor"]["terminal_semantic_scope"] == receipt["terminal_semantic_scope"] == scope
    assert set(loaded["semantic"]["manifest"]) == set(scope["selected_paths"])
    assert loaded["semantic"]["program_paths"] == case.selected
    assert set(loaded["indexed"]["source_sha256"]) == set(case.selected)
    assert loaded["semantic"]["raw_sources"][profiles.PROFILE].encode() == profiles.task_profile_bytes(case.declared)
    assert loaded["semantic"]["raw_sources"][case.selected[0]].encode() == case.source[case.selected[0]]
    assert scope["signature_authority"] is scope["omission_authority"] is False
    coverage = loaded["summaries"][1]["terminal_semantic_scope"]
    assert (coverage["selected_count"], coverage["full_count"], coverage["frontier_count"]) == (5, 73, 68)
    assert coverage["selected_scope_is_complete_repository"] is False
    assert "selected_sources" not in coverage and "sources" not in coverage
    semantic_path = case.root / loaded["descriptor"]["metadata"]["Semantic context artifact"]
    vector_path = case.root / ".runtime/terminal-vectors/result.json"
    originals = (semantic_path.read_bytes(), vector_path.read_bytes())
    admit(case)
    result = prep.context(state=case.state)
    assert result["initial_indexes_reused"] is True and result["new_embedding_calls"] == 0
    assert result["terminal_semantic_scope"] == scope
    assert result["index_id"] == receipt["index_id"] and result["semantic_root_cid"] == receipt["semantic_root_cid"]
    assert result["doctor_repair"]["automatic_repair_eligible"] is False
    assert result["doctor_repair"]["diagnostics_replayed"] is False
    assert (semantic_path.read_bytes(), vector_path.read_bytes()) == originals
    assert prep._load_prepared(case.state) == case.prepared
    assert {name:(case.root/name).read_bytes() for name in case.source} == case.source
    assert not (case.state / "planner-invoked.json").exists()


def test_fresh_admitted_context_without_initial_indexes_uses_the_same_signed_selection(case):
    admit(case)
    result = prep.context(state=case.state)
    assert result["initial_indexes_reused"] is False
    scope = result["terminal_semantic_scope"]
    assert scope == replay_terminal_semantic_scope(case.prepared)
    payload = json.loads((case.root/".runtime/terminal-context/semantic/worker-context.json").read_text())
    assert set(payload["manifest"]) == set(scope["selected_paths"])
    assert payload["program_paths"] == case.selected
    assert result["doctor_repair"]["complete_doctor_repair_evaluated"] is False
    assert len(case.prepared["manifest"]["payload"]["sources"]) == 73


@pytest.mark.parametrize("mutation", ["selected", "omitted", "support", "profile", "missing_prepared_profile", "unsigned_population"])
def test_complete_signed_custody_refuses_current_or_unsigned_scope_drift(case, mutation):
    prep.initial_context(state=case.state)
    prepared = deepcopy(case.prepared)
    if mutation in {"selected", "omitted", "support", "profile"}:
        path = {"selected":case.selected[0], "omitted":"module_069.py",
                "support":profiles.SMOKE, "profile":profiles.PROFILE}[mutation]
        (case.root/path).write_bytes((case.root/path).read_bytes()+b"\n")
    elif mutation == "missing_prepared_profile": prepared.pop("task_profile")
    else: prepared["worker_inputs"].remove("module_069.py")
    with pytest.raises(ValueError):
        initial.load_initial_context(state=case.state, prepared=prepared, require_empty_owner=True)
    assert not (case.state/"planner-invoked.json").exists()


def test_rehashed_unsigned_descriptor_cannot_substitute_a_different_selection(case):
    result = prep.initial_context(state=case.state)
    path = case.root/result["descriptor"]["artifact"]
    descriptor = json.loads(path.read_text())
    descriptor["terminal_semantic_scope"]["selected_paths"].append("module_069.py")
    prep._write(path, descriptor)
    result["descriptor"] = initial._reference(case.root, path)
    prep._write(case.state/"initial-context-result.json", result)
    with pytest.raises(ValueError):
        initial.load_initial_context(state=case.state, prepared=case.prepared, require_empty_owner=True)


def test_changed_omitted_source_during_final_empty_owner_capture_is_not_published(case, monkeypatch):
    original = initial.capture_intent_world_snapshot
    def change_during_capture(*args, **kwargs):
        result = original(*args, **kwargs)
        (case.root/"module_069.py").write_bytes(b"def changed(): return 0\n")
        return result
    monkeypatch.setattr(initial, "capture_intent_world_snapshot", change_during_capture)
    with pytest.raises(ValueError): prep.initial_context(state=case.state)
    assert not (case.state/"initial-context-result.json").exists()
    assert not (case.root/".runtime/terminal-initial-context/descriptor.json").exists()


@pytest.mark.parametrize("byte_delta", [0, 1])
def test_selected_original_byte_ceiling_counts_generated_support_exactly(tmp_path, byte_delta):
    case = make_case(tmp_path, byte_delta=byte_delta)
    if byte_delta:
        with pytest.raises(ValueError): replay_terminal_semantic_scope(case.prepared)
    else:
        receipt = replay_terminal_semantic_scope(case.prepared)
        assert receipt["selected_source_bytes"] == receipt["max_source_bytes"] == 1_000_000
        assert sum(row["source_bytes"] for row in receipt["selected_sources"].values()) == 1_000_000


@pytest.mark.parametrize("selected", ["training", "security", "source384", "numerical"])
def test_full_population_decoder_lanes_refuse_selection_before_context_outputs(case, selected):
    kwargs = {"training":{"train_autoencoder":True},"security":{"security_checkpoint":{}},
        "source384":{"source384_config":case.state/"missing.json"},
        "numerical":{"numerical_formalization_profile":"not-a-profile"}}[selected]
    with pytest.raises(ValueError, match="does not support decoder or training"):
        prep.initial_context(state=case.state, **kwargs)
    assert not (case.root/".runtime/terminal-vectors").exists()
    assert not (case.state/"initial-context-result.json").exists()


def test_independent_owner_cannot_sign_malformed_scoped_support(case, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
    malformed = deepcopy(case.declared)
    malformed["semantic_scope"]["proof_authority"] = True
    (case.root/profiles.PROFILE).write_text(json.dumps(malformed)+"\n")
    git(case.root,"add","--",profiles.PROFILE)
    git(case.root,"-c","user.name=New fixture owner","-c","user.email=scope@example.invalid","commit","-qm","Malformed declaration")
    profile_dir,lifecycle_dir = tmp_path/"new-profile",tmp_path/"new-lifecycle"
    Supervisor.init_local(repository=case.root,consent=True,profile_dir=profile_dir,lifecycle_dir=lifecycle_dir)
    specs = [deepcopy(case.prepared["spec"])]
    specs[0]["acceptance"][0]["evidence_cids"] = []
    with pytest.raises(ValueError):
        local.author_local_benchmark_manifest(repository=case.root,profile_dir=profile_dir,
            lifecycle_dir=lifecycle_dir,task_specs=specs,
            planning_roots=case.prepared["manifest"]["payload"]["planning_roots"])


def test_warm_native_owner_and_compiled_worker_keep_selected_scope_and_historical_audit(case):
    prep.initial_context(state=case.state)
    graph,admission = admit(case)
    context = prep.context(state=case.state)
    clone = case.state/"native-clone.duckdb"
    with IntentRepository(clone) as intent:
        local.materialize_local_benchmark_plan(admission=admission,intent=intent)
    task = graph.tasks[0]
    with open_existing_native_owner(database=clone,checkout=case.root,state_dir=case.state/"owner",
            repository_id=case.prepared["manifest"]["payload"]["repository_cid"],
            execution_routes={task.task_key:GROK_CODEX_EXECUTION_MODE}) as owner:
        before = owner.source.get_task(task.task_cid)
        rebound = rebind_full_context(prepared_state=case.state,admission=admission,
            server=owner.server,output=case.root/".runtime/rebound")
        assert rebound["terminal_semantic_scope"] == context["terminal_semantic_scope"]
        assert rebound["new_embedding_calls"] == rebound["text_generation_calls"] == 0
        portal = PortalTask(task_id=task.task_key,canonical_task_cid=task.task_cid,
            title=before.body["title"],status="ready",completion="manual",priority="P2",track="implementation",
            outputs=case.selected[:1],validation=[],acceptance="Public structure",metadata={"database task cid":task.task_cid})
        daemon = TodoImplementationDaemon(todo_path=case.root/".runtime/fixture.todo.md",
            state_path=case.state/"prompt/tasks.json",strategy_path=case.state/"prompt/strategy.json",
            events_path=case.state/"prompt/events.jsonl",repo_root=case.root,task_header_prefix="## TB-")
        daemon._task_context_nomination_bundle = rebound["context_bundle"]
        try:
            prompt = daemon._build_implementation_prompt(portal,attempt=1)
            observed = verify_worker_context_prompt(prompt=prompt,rebound=rebound)
            assert observed["task_completed"] is False and observed["completion_authority"] is False
            from benchmarks.agent_supervisor.container_coding import terminal_context_audit as audit
            daemon._persist_implementation_context_receipt(portal, 1)
            capsule = next((case.state/"prompt").rglob("*context-capsule.json"))
            # Historical audit must not need a surviving checkout or live source.
            (case.root/"module_069.py").write_bytes(b"def later_state(): return 7\n")
            historical,rendered,_ = audit._capsule(capsule.read_bytes(),expected=rebound,prepared=case.prepared)
            assert historical["all_context_checks_passed"] is True
            assert historical["scope_observation_is_historical_only"] is True
            assert historical["all_selected_lengths_independently_replayed"] is False
            assert verify_worker_context_prompt(prompt=prompt,rebound=rebound) == observed
            changed = deepcopy(case.prepared);changed.pop("task_profile")
            altered = deepcopy(rebound);altered.pop("terminal_semantic_scope")
            with pytest.raises(ValueError): audit._capsule(capsule.read_bytes(),expected=altered,prepared=changed)
            with pytest.raises(ValueError):
                rebind_full_context(prepared_state=case.state,admission=admission,
                    server=owner.server,output=case.root/".runtime/rebound-again")
            assert owner.source.get_task(task.task_cid) == before
        finally: daemon.close_event_runtime()
