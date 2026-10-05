"""Absence successors retain native scope and cannot select an embedding lane."""
from copy import deepcopy
from contextlib import ExitStack
from dataclasses import replace
import hashlib
import subprocess
from types import SimpleNamespace

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario  # noqa: F401
from test.api.test_local_completion_bridge import typed_claim, native_published_transition, complete
from test.api.test_local_planning_declared_create import _with_creation
from test.api.semantic_state.test_published_task_context import initial_bundle, vectors
from benchmarks.agent_supervisor.container_coding.terminal_container_supervisor import (
    _published_retrieval_options, _empty_retrieval_activity,
)
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import Supervisor
from ipfs_accelerate_py.agent_supervisor.runtime import published_task_context as runtime
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import prepare_code_retrieval_context
from ipfs_accelerate_py.agent_supervisor.runtime.empty_code_retrieval import (
    prepare_empty_code_retrieval_context, read_empty_code_retrieval_artifact,
)
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE


def support(root, names):
    return {name: {"role": role, "sha256": hashlib.sha256((root / name).read_bytes()).hexdigest()}
        for name, role in names.items()}


def empty_predecessor(tmp_path, *, program=True):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "instruction.md").write_text("Create the declared output without expanding old program scope.\n")
    paths = ["answer.py"] if program else []
    if program:
        (root / "answer.py").write_text("answer = 1\n")
    prepared = prepare_empty_code_retrieval_context(repository=root, task_id="TASK", query_text="answer",
        program_paths=paths, support_hashes=support(root, {"instruction.md": "instruction"}),
        output=root / ".runtime/initial-retrieval.json")
    task_cid = local.content_identity({"task": "empty-publication-test"})
    bundle = write_task_context_bundle(repository=root, prepared=[{
        "schema": "supervisor-task-context-preparation@1", "task_cid": task_cid,
        "task_id": "TASK", "metadata": prepared["metadata"]}], output=root / ".runtime/bundle.json")
    payload, context = read_empty_code_retrieval_artifact(repository=root,
        artifact=prepared["metadata"]["Code retrieval artifact"],
        expected_sha256=prepared["metadata"]["Code retrieval sha256"], task_id="TASK")
    return SimpleNamespace(root=root, prepared=prepared, payload=payload, context=context,
        task=SimpleNamespace(task_cid=task_cid, task_key="TASK"), bundle=bundle)


@pytest.mark.parametrize("program,change", [(False, "unchanged"), (False, "support"),
    (True, "unchanged"), (True, "comment"), (True, "support"), (True, "value"), (True, "symbol")])
def test_empty_successor_reobserves_same_scope_without_model(tmp_path, program, change):
    original = empty_predecessor(tmp_path, program=program)
    metadata = {key.lower(): value for key, value in original.prepared["metadata"].items()}
    original_bytes = (original.root / metadata["code retrieval artifact"]).read_bytes()
    if change == "support":
        (original.root / "instruction.md").write_text("Changed accepted support bytes.\n")
    elif change == "comment":
        (original.root / "answer.py").write_text("# Changed source comment\nanswer = 1\n")
    elif change == "value":
        (original.root / "answer.py").write_text("answer = 2\n")
    elif change == "symbol":
        (original.root / "answer.py").write_text("def answer():\n    return 2\n")
    # A new code output is outside the authenticated old scope even if it has symbols.
    (original.root / "new_module.py").write_text("def created_output():\n    return 42\n")
    context, snapshot, payload = runtime._previous_retrieval(original.root, metadata, "TASK")
    assert snapshot is None and payload == original.payload
    successor, observed = runtime._refresh_empty_retrieval(root=original.root, metadata=metadata,
        context=context, payload=payload, task_id="TASK", output=original.root / ".runtime/successor",
        declared_outputs=["new_module.py"])
    assert observed["embedding_calls"] == observed["model_loading_calls"] == 0
    assert observed["index_id"] is None
    assert observed["preserved_program_paths"] == (["answer.py"] if program else [])
    assert observed["preserved_support_roles"] == {"instruction.md": "instruction"}
    assert observed["declared_outputs_outside_index_scope"] == ["new_module.py"]
    assert (original.root / metadata["code retrieval artifact"]).read_bytes() == original_bytes
    if change == "symbol":
        assert observed["status"] == "unavailable" and successor == {}
        assert observed["reason"] == "qualified_symbols_require_independent_vector_policy"
        assert observed["source_population_cid"] is None
    else:
        current, worker = read_empty_code_retrieval_artifact(repository=original.root,
            artifact=successor["Code retrieval artifact"], expected_sha256=successor["Code retrieval sha256"],
            task_id="TASK")
        assert observed["status"] == worker["status"] == "current" and worker["hits"] == []
        assert observed["source_population_cid"] == current["source_population_cid"]
        assert current["program_paths"] == original.payload["program_paths"]
        assert current["support_hashes"] == support(original.root, {"instruction.md": "instruction"})
        if change == "unchanged":
            assert successor == original.prepared["metadata"]
            assert observed["source_population_cid"] == original.payload["source_population_cid"]
        else:
            assert observed["source_population_cid"] != original.payload["source_population_cid"]


def test_empty_publication_policy_is_closed_and_retains_historical_scope(tmp_path):
    original = empty_predecessor(tmp_path)
    binding = runtime.bind_published_empty_retrieval_policy(repository=original.root, bundle=original.bundle,
        task_cid=original.task.task_cid, task_id=original.task.task_key)
    assert binding["policy"] == runtime.EMPTY_RETRIEVAL_POLICY and binding["index_id"] is None
    assert binding["embedding_calls"] == 0
    (original.root / "answer.py").write_text("def answer():\n    return 2\n")
    assert runtime.validate_published_empty_retrieval_policy(repository=original.root,
        bundle=original.bundle, binding=binding) == binding
    for field, value in [("program_paths", []), ("support_roles", {}), ("source_population_cid", "forged"),
            ("embedding_calls", 1), ("execution_authority", True), ("policy", "lexical-tfidf-symbols@1"),
            ("invented_field", True)]:
        with pytest.raises(ValueError):
            runtime.validate_published_empty_retrieval_policy(repository=original.root,
                bundle=original.bundle, binding={**binding, field: value})


def test_container_empty_policy_ignores_unused_selected_snapshot_and_rejects_stale_scope(tmp_path):
    original = empty_predecessor(tmp_path)
    options = _published_retrieval_options(repository=original.root, bundle=original.bundle,
        task=original.task, model_snapshot=tmp_path / "never-read-model")
    assert options == dict(published_retrieval_policy=runtime.EMPTY_RETRIEVAL_POLICY,
        published_learned_artifacts=None)
    (original.root / "answer.py").write_text("answer = 2\n")
    with pytest.raises(ValueError, match="current"):
        _published_retrieval_options(repository=original.root, bundle=original.bundle,
            task=original.task, model_snapshot=None)


def test_normal_vector_predecessor_keeps_native_snapshot_result_union(tmp_path):
    root = tmp_path / "repository"
    root.mkdir()
    (root / "answer.py").write_text("def answer():\n    return 1\n")
    snapshot, hits = vectors(root)
    prepared = prepare_code_retrieval_context(repository=root, task_id="TASK", query_text="answer",
        snapshot=snapshot, result=hits, output=root / ".runtime/retrieval.json")
    metadata = {key.lower(): value for key, value in prepared["metadata"].items()}
    context, checked_snapshot, checked_result = runtime._previous_retrieval(root, metadata, "TASK")
    assert context["status"] == "current" and checked_snapshot == snapshot and checked_result == hits


def test_empty_activity_is_distinct_closed_zero_model_observation(tmp_path):
    original = empty_predecessor(tmp_path)
    observed = dict(schema="supervisor-published-empty-retrieval-observation@1", status="current",
        source_population_cid=original.payload["source_population_cid"],
        previous_source_population_cid=original.payload["source_population_cid"],
        embedding_calls=0, model_loading_calls=0, execution_authority=False, completion_authority=False)
    assert _empty_retrieval_activity(observed) == dict(local_embedding_calls=0, local_embedding_texts=0,
        remote_embedding_calls=0, text_generation_calls=0)
    for field, value in [("schema", "supervisor-published-learned-embedding-receipt@1"),
            ("embedding_calls", 1), ("model_loading_calls", False), ("execution_authority", True),
            ("source_population_cid", None), ("previous_source_population_cid", "forged"), ("extra", True)]:
        assert _empty_retrieval_activity({**observed, field: value}) is None


def zero_symbol_scenario(scenario):
    root = scenario["repository"]
    (root / "answer.py").write_text("answer = 1\n")
    (root / "test_answer.py").write_text(
        "from answer import answer\nassert (answer() if callable(answer) else answer) == 2\n")
    subprocess.run(["git", "-C", str(root), "add", "answer.py", "test_answer.py"], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Empty scope qualification", "-c",
        "user.email=empty@example.invalid", "commit", "-qm", "Authored zero-symbol public task"], check=True)
    roots = deepcopy(scenario["manifest"]["payload"]["planning_roots"])
    roots["scan_cid"] = local.content_identity({"sources": local._sources(root, ["answer.py", "test_answer.py"])})
    roots["program_root"] = local.content_identity({"git_tree": subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD^{tree}"], text=True).strip()})
    graph = replace(scenario["graph"], **roots)
    declared = scenario["manifest"]["payload"]
    profile = root.parent / "zero-symbol-profile"
    lifecycle = root.parent / "zero-symbol-lifecycle"
    Supervisor.init_local(repository=root, consent=True, profile_dir=profile, lifecycle_dir=lifecycle)
    manifest = local.author_local_benchmark_manifest(repository=root, profile_dir=profile,
        lifecycle_dir=lifecycle, task_specs=deepcopy(declared["tasks"]), planning_roots=roots)
    return {**scenario, "graph": graph, "manifest": manifest, "task_cid": graph.tasks[0].task_cid,
        "profile": profile, "lifecycle": lifecycle}


@pytest.mark.parametrize("publication", ["zero_symbol", "first_symbol", "created_code"])
def test_actual_completed_empty_publication_preserves_scope_and_current_owner(scenario, tmp_path, publication):
    case = zero_symbol_scenario(scenario)
    if publication == "created_code":
        graph, manifest = _with_creation(case, name="new_module.py")
        case = {**case, "graph": graph, "manifest": manifest, "task_cid": graph.tasks[0].task_cid}
    admission = local.admit_local_benchmark_plan(graph=case["graph"], manifest=case["manifest"])
    root = case["repository"]
    with typed_claim(case, tmp_path) as (owner, _daemon, attempt), ExitStack() as cleanup:
        population = dict(program_paths=["answer.py"],
            support_hashes=support(root, {"test_answer.py": "structural_smoke"}))
        bundle = initial_bundle(case, owner, empty_population=population)
        old_bytes = (root / bundle["artifact"]).read_bytes()
        admitted = None
        if publication == "zero_symbol":
            admitted = AdmittedBenchmarkRuntime.create(tmp_path / "published-empty-launch", admission=admission,
                server=owner.server, source=owner.source, context_bundle=bundle,
                refresh_context_on_completion=True, published_retrieval_policy=runtime.EMPTY_RETRIEVAL_POLICY)
            cleanup.callback(admitted.close)
        source = "def answer():\n    return 2\n" if publication == "first_symbol" else "# Accepted repair\nanswer = 2\n"
        transition = native_published_transition(case, tmp_path, owner, attempt,
            modified_outputs={"answer.py": source},
            created_outputs={"new_module.py": "def created_output():\n    return 42\n"}
                if publication == "created_code" else None)
        task = owner.source.get_task(attempt.task_cid)
        passed = run_owner_local_task_validations(server=owner.server, task_cid=attempt.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
        assert passed["passed"]
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        canonical = owner.source.get_task(attempt.task_cid)
        if publication == "zero_symbol":
            with pytest.raises(ValueError, match="empty retrieval cannot select a vector producer"):
                runtime.refresh_published_task_context(server=owner.server, admission=admission,
                    predecessor_bundle=bundle, task_cid=attempt.task_cid,
                    output=root / ".runtime/forbidden-vector-transition",
                    retrieval_rebuilder=lambda **kwargs: pytest.fail("empty successor called embedding producer"))
        refreshed = runtime.refresh_published_task_context(server=owner.server, admission=admission,
            predecessor_bundle=bundle, task_cid=attempt.task_cid, output=root / ".runtime/published-empty")
        loaded = runtime.load_published_task_context(server=owner.server, admission=admission,
            artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"])
        retrieval = loaded["retrieval"]
        assert retrieval["embedding_calls"] == retrieval["model_loading_calls"] == 0 and retrieval["index_id"] is None
        assert retrieval["preserved_program_paths"] == ["answer.py"]
        assert retrieval["preserved_support_roles"] == {"test_answer.py": "structural_smoke"}
        assert loaded["intent_freshness_checked"] and loaded["task_status"] == "completed"
        assert loaded["completion_authority"] is False and loaded["execution_authority"] is False
        assert owner.source.get_task(attempt.task_cid) == canonical
        assert (root / bundle["artifact"]).read_bytes() == old_bytes
        if publication == "first_symbol":
            assert retrieval["status"] == "unavailable"
            assert retrieval["reason"] == "qualified_symbols_require_independent_vector_policy"
            assert "Code retrieval artifact" not in loaded["metadata"]
        else:
            assert retrieval["status"] == "current" and retrieval["disposition"] == "zero_qualified_symbols"
            assert retrieval["source_population_cid"] != retrieval["previous_source_population_cid"]
        if publication == "created_code":
            assert retrieval["declared_outputs_outside_index_scope"] == ["new_module.py"]
        forged = deepcopy(refreshed)
        if retrieval["status"] == "current":
            forged["retrieval"]["source_population_cid"] = retrieval["previous_source_population_cid"]
        else:
            forged["retrieval"]["source_sha256"]["answer.py"] = "0" * 64
        forged_path = root / ".runtime/published-empty/forged-result.json"
        forged_bytes = runtime._json(forged).encode()
        forged_path.write_bytes(forged_bytes)
        with pytest.raises(ValueError, match="empty retrieval.*(identity|transition)"):
            runtime.load_published_task_context(server=owner.server, admission=admission,
                artifact=forged_path.relative_to(root).as_posix(),
                expected_sha256=hashlib.sha256(forged_bytes).hexdigest())
        if admitted is not None:
            stopped = admitted.stop()
            assert stopped.succeeded and stopped.data["old_tree_fenced"] is True
            observations = admitted.refresh_after_stop()
            assert len(observations) == 1 and observations[0]["status"] == "refreshed", observations
            assert observations[0]["retrieval_status"] == "current"
            activity = _empty_retrieval_activity(observations[0]["empty_retrieval_observation"])
            assert activity is not None and all(value == 0 for value in activity.values())
            assert "embedding_receipt" not in observations[0]
            cached = admitted.refresh_after_stop()
            assert cached[0]["refresh_sha256"] == observations[0]["refresh_sha256"]
        (root / "test_answer.py").write_text("assert True\n")
        with pytest.raises(local.LocalPlanningError):
            runtime.load_published_task_context(server=owner.server, admission=admission,
                artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"])


def test_admitted_owner_registers_empty_policy_without_vector_artifacts(scenario, tmp_path):
    case = zero_symbol_scenario(scenario)
    admission = local.admit_local_benchmark_plan(graph=case["graph"], manifest=case["manifest"])
    root = case["repository"]
    local.materialize_local_benchmark_plan(admission=admission, intent=case["intent"])
    with open_existing_native_owner(database=case["intent"].database_path, checkout=root,
            state_dir=tmp_path / "empty-admitted-owner", repository_id=case["manifest"]["payload"]["repository_cid"],
            execution_routes={case["graph"].tasks[0].task_key: GROK_CODEX_EXECUTION_MODE}) as owner:
        bundle = initial_bundle(case, owner, empty_population=dict(program_paths=["answer.py"],
            support_hashes=support(root, {"test_answer.py": "structural_smoke"})))
        admitted = AdmittedBenchmarkRuntime.create(tmp_path / "empty-launch", admission=admission,
            server=owner.server, source=owner.source, context_bundle=bundle,
            refresh_context_on_completion=True, published_retrieval_policy=runtime.EMPTY_RETRIEVAL_POLICY)
        try:
            binding = admitted.manifest["published_retrieval_policy"]
            assert binding["policy"] == runtime.EMPTY_RETRIEVAL_POLICY and binding["index_id"] is None
            assert binding["program_paths"] == ["answer.py"] and binding["embedding_calls"] == 0
            assert not admitted.process.snapshot(admitted.profile).members
        finally:
            admitted.close()
        with pytest.raises(ValueError, match="learned refresh"):
            AdmittedBenchmarkRuntime.create(tmp_path / "wrong-launch", admission=admission,
                server=owner.server, source=owner.source, context_bundle=bundle,
                refresh_context_on_completion=True, published_retrieval_policy=runtime.EMPTY_RETRIEVAL_POLICY,
                published_learned_artifacts={"model_snapshot": "/unused"})
