"""Real completed publication retains bounded immutable v2 task-data references."""
import json

import pytest

from benchmarks.agent_supervisor.container_coding.test_terminal_data_profiles import mixed
from benchmarks.agent_supervisor.container_coding.test_terminal_task_profile import prepare
from benchmarks.agent_supervisor.container_coding.test_terminal_indexed_preparation import _proposal_json
from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as preparation
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanReceipt, PromptWorkflowRequest,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import parse_prompt_goal_graph
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import terminal_task_profile as profiles
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.runtime.published_task_context import (
    refresh_published_task_context, load_published_task_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import (
    load_semantic_worker_context, prepare_semantic_context,
)
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import write_task_context_bundle
from ipfs_accelerate_py.agent_supervisor.semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot, persist_intent_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from test.api.test_local_completion_bridge import typed_claim, native_published_transition, complete


@pytest.fixture
def data_case(tmp_path):
    args = mixed(tmp_path, media="application/x-ndjson", name="requests.jsonl",
        raw=b'{"request_id":"example","size":128}\n' * 2800)
    root, instruction, _, profile = args
    instruction.write_text("Update source.py using immutable requests.jsonl.\n")
    profile.update(instruction_sha256=profiles.instruction_sha256(instruction.read_text()),
        outputs=[dict(path="source.py", effect="modify", media_type="text/x-python")])
    prepared = prepare(args)
    proposal = json.loads(_proposal_json(prepared))
    proposal["tasks"][0]["predicted_files"] = ["source.py"]
    # Authored parser fixture: no provider or autoencoder inference is invoked.
    graph = parse_prompt_goal_graph(json.dumps(proposal),
        PromptWorkflowRequest.from_dict(prepared["request"]),
        DirectoryScanReceipt.from_dict(prepared["scan"]),
        config=preparation._config(root), constraint_summaries=prepared["constraints"])
    with IntentRepository(tmp_path / "intent.duckdb") as intent:
        yield dict(repository=root, graph=graph, manifest=prepared["manifest"], intent=intent,
            task_cid=graph.tasks[0].task_cid, prepared=prepared)


def predecessor(case, owner):
    root = case["repository"]
    alias = case["graph"].tasks[0].task_key
    out = root / ".runtime/predecessor"
    semantic = prepare_semantic_context(repository=root,
        paths=case["prepared"]["worker_inputs"], required_raw_paths=[profiles.INSTRUCTION, profiles.SMOKE],
        program_paths=["source.py"], defer_task_data=True, worker_query="Update source.py",
        objective="Update source.py", task_id=alias, output=out / "semantic")
    get_block = lambda cid: (out / "semantic/blocks" / cid).read_bytes()
    view = IpfsDatasetsSemanticStateProvider().open_verified_view(semantic["semantic_root_cid"], get_block)
    with owner.server._lock:
        connection = owner.server._connection
        connection.execute("BEGIN TRANSACTION")
        try:
            capture = capture_intent_world_snapshot(
                IntentRepository(bound_connection=connection, install_schema=False),
                repository_id=view.root.repository_id, task_cids=[case["task_cid"]],
                semantic_root_cid=semantic["semantic_root_cid"], get_semantic_block=get_block,
                transaction_owned_by_caller=True)
            connection.execute("COMMIT")
        except BaseException:
            connection.execute("ROLLBACK")
            raise
    world = persist_intent_world_snapshot(capture, output=out / "world", task_id=alias)
    metadata = {
        "Semantic context artifact": (out / "semantic/worker-context.json").relative_to(root).as_posix(),
        "Semantic context sha256": semantic["worker_payload_sha256"],
        "Semantic context refresh": "true",
        "World context artifact": (out / "world/intent-world.json").relative_to(root).as_posix(),
        "World context sha256": world["artifact_sha256"],
        "World context repository": view.root.repository_id,
    }
    bundle = write_task_context_bundle(repository=root, prepared=[{
        "schema": "supervisor-task-context-preparation@1", "task_cid": case["task_cid"],
        "task_id": alias, "metadata": metadata}], output=out / "bundle.json")
    return bundle, json.loads((out / "semantic/worker-context.json").read_bytes())


@pytest.mark.parametrize("drift", [None, "requests.jsonl", profiles.INSTRUCTION, profiles.PROFILE, profiles.SMOKE])
def test_completed_publication_preserves_deferred_bytes_and_refuses_immutable_drift(data_case, tmp_path, drift):
    case = data_case
    root = case["repository"]
    admission = local.admit_local_benchmark_plan(graph=case["graph"], manifest=case["manifest"])
    with typed_claim(case, tmp_path) as (owner, _daemon, attempt):
        bundle, previous = predecessor(case, owner)
        original_bundle = (root / bundle["artifact"]).read_bytes()
        projection = previous["task_data_projection"]
        data_reference = projection["raw_source_fetch_required"]["requests.jsonl"]
        assert len((root / "requests.jsonl").read_bytes()) > 90000
        assert "requests.jsonl" not in previous["raw_sources"]
        transition = native_published_transition(case, tmp_path, owner, attempt,
            modified_outputs={"source.py": "def public_source():\n    return 2\n"})
        task = owner.source.get_task(attempt.task_cid)
        passed = run_owner_local_task_validations(server=owner.server, task_cid=attempt.task_cid,
            attempt_id=attempt.attempt_id, expected_revision=task.revision, source_transition=transition)
        assert passed["passed"]
        complete(owner, attempt, passed["results"][0]["evidence_digest"])
        canonical = owner.source.get_task(attempt.task_cid)
        if drift is not None:
            path = root / drift
            path.write_bytes(path.read_bytes().replace(b"128", b"129") if drift == "requests.jsonl"
                else path.read_bytes() + b"\n")
        output = root / ".runtime/published"
        arguments = dict(server=owner.server, admission=admission, predecessor_bundle=bundle,
            task_cid=attempt.task_cid, output=output)
        if drift is not None:
            with pytest.raises((ValueError, local.LocalPlanningError)):
                refresh_published_task_context(**arguments)
            assert not output.exists()
        else:
            refreshed = refresh_published_task_context(**arguments)
            loaded = load_published_task_context(server=owner.server, admission=admission,
                artifact=refreshed["refresh_artifact"], expected_sha256=refreshed["refresh_sha256"])
            assert loaded["task_status"] == "completed"
            metadata = loaded["metadata"]
            text = load_semantic_worker_context(repository=root,
                artifact=metadata["Semantic context artifact"],
                expected_sha256=metadata["Semantic context sha256"], task_id=case["graph"].tasks[0].task_key)
            successor = json.loads(text)
            assert len(text.encode()) <= 32768
            assert successor["task_data_projection"] == projection
            assert successor["raw_sources"] == previous["raw_sources"]
            assert successor["program_paths"] == previous["program_paths"] == ["source.py"]
            assert successor["semantic_root_cid"] != previous["semantic_root_cid"]
            assert set(successor["refresh_lineage"]["source_delta"]) == {"source.py"}
            assert successor["refresh_lineage"]["cause"] == "owner_validated_publication_completed"
            assert successor["worker_projection"]["raw_source_fetch_required"]["requests.jsonl"] == previous["manifest"]["requests.jsonl"]
            assert (output / "semantic/blocks" / data_reference["source_cid"]).read_bytes() == (root / "requests.jsonl").read_bytes()
            assert loaded["completion_authority"] is False
        assert (root / bundle["artifact"]).read_bytes() == original_bundle
        assert owner.source.get_task(attempt.task_cid) == canonical
