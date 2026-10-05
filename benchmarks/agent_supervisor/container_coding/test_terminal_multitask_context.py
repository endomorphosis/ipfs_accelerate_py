"""Ready root tasks reuse real retained vectors in task-bound context evidence.

The reviewed three-task inputs are authored fixtures. Source capsules, numeric
retrieval replay, intent worlds, native task readiness and context consumers run
their real implementations. Context evidence grants no launch authority.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json

import duckdb
import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_multitask_context as contexts
from benchmarks.agent_supervisor.container_coding import vector_index_preflight
from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from benchmarks.agent_supervisor.container_coding.test_terminal_multitask_preparation import (
    multitask_case, _prepare,  # noqa: F401
)
from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
    CodeVectorIndexSnapshot, CodeVectorSearchResult,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import PromptGoalGraph
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
from ipfs_accelerate_py.agent_supervisor.runtime.code_retrieval_context import load_code_retrieval_context
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import load_semantic_worker_context
from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import (
    load_task_context_nomination, write_task_context_bundle,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import (
    capture_intent_world_snapshot, load_intent_world_context, persist_intent_world_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def admitted(multitask_case):
    case = multitask_case
    prepared = _prepare(case)
    planned = prep.plan(case["state"])
    assert planned["qualified"] is True, planned
    admission = json.loads((case["state"] / "admission.json").read_bytes())
    tasks = {task.task_key: task for task in PromptGoalGraph.from_dict(admission["graph"]).tasks}
    return {**case, "prepared": prepared, "admission": admission, "tasks": tasks,
            "roots": sorted(tasks[key].task_cid for key in ("TB-LEFT", "TB-RIGHT")),
            "output": case["repository"] / ".runtime/ready-task-contexts"}


@pytest.fixture
def retained_vectors(admitted, tmp_path):
    """Create and hydrate a genuine lexical index before the tested operation."""
    root = admitted["repository"]
    output = tmp_path / "retained-vector-assets"
    qualification = vector_index_preflight.qualify(root, output,
        admitted["profile"]["input_paths"], admitted["prepared"]["query"])
    with duckdb.connect(str(output / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
        snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
            "SELECT payload FROM snapshots WHERE id=?", [qualification["index_id"]]).fetchone()[0]))
    result = CodeVectorSearchResult.from_dict(qualification["hits"])
    files = [output / name for name in ("vectors.duckdb", "ast.json", "evidence.json", "result.json")]
    return {"snapshot": snapshot, "result": result, "files": files,
            "hashes": {str(path): _hash(path) for path in files},
            "snapshot_bytes": _wire(snapshot.to_dict()), "result_bytes": _wire(result.to_dict())}


def _prepare_contexts(case, vectors=None, *, selected=None, output=None):
    options = {} if vectors is None else {"code_vector_snapshot": vectors["snapshot"],
                                         "code_vector_result": vectors["result"]}
    return contexts.prepare_ready_task_contexts(state=case["state"],
        task_cids=case["roots"] if selected is None else selected,
        output=case["output"] if output is None else output, **options)


def _assert_retained(vectors):
    assert {str(path): _hash(path) for path in vectors["files"]} == vectors["hashes"]
    assert _wire(vectors["snapshot"].to_dict()) == vectors["snapshot_bytes"]
    assert _wire(vectors["result"].to_dict()) == vectors["result_bytes"]


def _cold(case, *, artifact=None, digest=None):
    receipt = case["output"] / "result.json"
    return contexts.load_ready_task_contexts(state=case["state"],
        artifact=receipt.relative_to(case["repository"]).as_posix() if artifact is None else artifact,
        expected_sha256=_hash(receipt) if digest is None else digest)


def _rewrite_receipt(case, result):
    """Seal a caller-authored receipt; native joins must still be verified."""
    receipt = case["output"] / "result.json"
    receipt.write_bytes(_wire(result))
    return _hash(receipt)


def test_two_ready_roots_reuse_vectors_and_bind_actual_semantic_world_evidence(admitted, retained_vectors, tmp_path):
    case, vectors = admitted, retained_vectors
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        before, watermark = intent.plan_projection(), intent.event_watermark()
        assert {row["task_cid"] for row in intent.select_ready_tasks()} == set(case["roots"])
    result = _prepare_contexts(case, vectors)
    assert result["schema"] == "terminal-reviewed-ready-task-contexts@1"
    assert result["task_cids"] == case["roots"]
    assert len(result["contexts"]) == 2
    for key in ("execution_authority", "completion_authority"):
        assert result[key] is False
    for key in ("new_embedding_calls", "model_loading_calls", "training_steps", "provider_calls"):
        assert result[key] == 0
    assert result["query_semantic_alignment_verified"] is False
    bundle = result["context_bundle"]
    aliases = set()
    world_paths = set()
    semantic_paths = set()
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        assert intent.plan_projection() == before and intent.event_watermark() == watermark
        for item in result["contexts"]:
            assert item["task_cid"] in case["roots"]
            assert item["execution_authority"] is item["completion_authority"] is False
            assert item["canonical_task_mutated"] is False
            aliases.add(item["task_id"])
            metadata = load_task_context_nomination(repository=case["repository"], artifact=bundle["artifact"],
                expected_sha256=bundle["sha256"], task_cid=item["task_cid"], task_id=item["task_id"])
            semantic = json.loads(load_semantic_worker_context(repository=case["repository"],
                artifact=metadata["semantic context artifact"],
                expected_sha256=metadata["semantic context sha256"], task_id=item["task_id"]))
            world = load_intent_world_context(artifact=case["repository"] / metadata["world context artifact"],
                expected_sha256=metadata["world context sha256"], task_id=item["task_id"],
                repository_id=metadata["world context repository"], intent=intent)
            retrieval = json.loads(load_code_retrieval_context(repository=case["repository"],
                artifact=metadata["code retrieval artifact"], expected_sha256=metadata["code retrieval sha256"],
                task_id=item["task_id"]))
            assert world["intent_freshness_checked"] is True
            assert world["semantic_root_cid"] == semantic["semantic_root_cid"] == item["semantic_root_cid"]
            assert {row["task_cid"] for row in world["tasks"]} == {item["task_cid"]}
            assert semantic["program_paths"] == case["profile"]["input_paths"]
            assert semantic["raw_sources"][prep.INSTRUCTION] == case["prepared"]["query"]
            assert retrieval["status"] == "current" and retrieval["task_id"] == item["task_id"]
            assert retrieval["index_id"] == vectors["snapshot"].index_id
            assert retrieval["result_id"] == vectors["result"].result_id
            assert retrieval["query_id"] == vectors["result"].query.query_id
            world_paths.add(metadata["world context artifact"])
            semantic_paths.add(metadata["semantic context artifact"])
    assert aliases == {"TB-LEFT", "TB-RIGHT"}
    assert len(world_paths) == len(semantic_paths) == 2
    assert _cold(case) == result
    _assert_retained(vectors)
    with open_existing_native_owner(database=case["state"] / "intent.duckdb", checkout=case["repository"],
        state_dir=tmp_path / "native-owner", repository_id=case["prepared"]["manifest"]["payload"]["repository_cid"],
        execution_routes={key: GROK_CODEX_EXECUTION_MODE for key in case["tasks"]}) as owner:
        assert {row.task_cid for row in owner.source.ready_tasks().tasks} == set(case["roots"])
        with owner.server._lock:
            intent = IntentRepository(bound_connection=owner.server._connection, install_schema=False)
            for item in result["contexts"]:
                metadata = item["metadata"]
                world = load_intent_world_context(artifact=case["repository"] / metadata["World context artifact"],
                    expected_sha256=metadata["World context sha256"], task_id=item["task_id"],
                    repository_id=metadata["World context repository"], intent=intent)
                assert world["intent_freshness_checked"]
        for item in result["contexts"]:
            native = owner.source.get_task(item["task_cid"])
            assert native.task_alias == item["task_id"] and native.status == "ready"
        launch = tmp_path / "unallocated-launch"
        with pytest.raises(ValueError, match="administrative"):
            AdmittedBenchmarkRuntime.create(launch, admission=case["admission"], server=owner.server,
                source=owner.source, context_bundle=bundle)
        assert not launch.exists()
    assert owner.server.status()["lifecycle"] == "stopped"


@pytest.mark.parametrize("selection", ["unknown", "alias", "duplicate", "dependent", "empty"])
def test_selection_refusals_precede_context_outputs(admitted, selection):
    case = admitted
    selected = {"unknown": ["task:foreign"], "alias": ["TB-LEFT"],
        "duplicate": [case["roots"][0], case["roots"][0]],
        "dependent": [case["tasks"]["TB-JOIN"].task_cid], "empty": []}[selection]
    with pytest.raises((ValueError, KeyError)):
        _prepare_contexts(case, selected=selected)
    assert not case["output"].exists()


def test_actual_foreign_admitted_task_is_not_a_local_context_selection(admitted, tmp_path):
    foreign_root = tmp_path / "foreign-case"
    foreign_root.mkdir()
    foreign = multitask_case.__wrapped__(foreign_root)
    _prepare(foreign)
    assert prep.plan(foreign["state"])["qualified"]
    foreign_admission = json.loads((foreign["state"] / "admission.json").read_bytes())
    foreign_task = PromptGoalGraph.from_dict(foreign_admission["graph"]).tasks[0].task_cid
    assert foreign_task not in {task.task_cid for task in admitted["tasks"].values()}
    with pytest.raises(ValueError, match="absent"):
        _prepare_contexts(admitted, selected=[foreign_task])
    assert not admitted["output"].exists()


@pytest.mark.parametrize("mutation", ["body", "dependencies", "validations", "owner", "alias", "blocked"])
def test_actual_native_row_changes_cannot_enter_task_context(admitted, retained_vectors, mutation):
    """Corrupt actual stored rows to test independent owner verification."""
    case = admitted
    selected = case["tasks"]["TB-LEFT"].task_cid
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        row = intent.get_task(selected)
        with intent._connection(write=True) as connection:
            if mutation in {"body", "owner"}:
                body = deepcopy(row["body"])
                if mutation == "body":
                    body["title"] = "Substituted task objective"
                else:
                    body[local.CONTRACT_KEY]["payload"]["intent_owner_id"] = "foreign-owner"
                connection.execute("UPDATE tasks SET body_json=? WHERE task_cid=?",
                                   [_wire(body).decode(), selected])
            elif mutation == "dependencies":
                connection.execute("INSERT INTO task_dependencies(task_cid, dependency_task_cid, kind) VALUES(?, ?, ?)",
                                   [selected, case["tasks"]["TB-RIGHT"].task_cid, "depends_on"])
            elif mutation == "validations":
                connection.execute("UPDATE task_validations SET argv_json=? WHERE task_cid=?",
                                   ['["python3","-c","pass"]', selected])
            elif mutation == "alias":
                connection.execute("UPDATE tasks SET task_alias=? WHERE task_cid=?", ["FOREIGN-ALIAS", selected])
            else:
                connection.execute("UPDATE tasks SET status=? WHERE task_cid=?", ["blocked", selected])
    with pytest.raises(ValueError):
        _prepare_contexts(case, retained_vectors, selected=[selected])
    assert not case["output"].exists()


@pytest.mark.parametrize("path", ["left.py", prep.INSTRUCTION, prep.SMOKE, ".supervisor-task-profile.json"])
def test_changed_signed_source_refused_before_context_publication(admitted, path):
    case = admitted
    selected = case["repository"] / path
    selected.write_bytes(selected.read_bytes() + b"\n# changed actual source\n")
    with pytest.raises(ValueError):
        _prepare_contexts(case)
    assert not case["output"].exists()


@pytest.mark.parametrize("mutation", ["query", "snapshot", "source", "missing_pair"])
def test_retained_numeric_binding_refusals_do_not_publish_contexts(admitted, retained_vectors, mutation):
    case, vectors = admitted, retained_vectors
    options = dict(code_vector_snapshot=vectors["snapshot"], code_vector_result=vectors["result"])
    if mutation == "query":
        options["code_vector_result"] = replace(vectors["result"], hits=())
    elif mutation == "snapshot":
        options["code_vector_snapshot"] = replace(vectors["snapshot"],
            config=replace(vectors["snapshot"].config, model_revision="substituted-revision"))
    elif mutation == "source":
        (case["repository"] / "left.py").write_text("def left():\n    return 'new source'\n")
    else:
        options["code_vector_result"] = None
    with pytest.raises((ValueError, TypeError)):
        contexts.prepare_ready_task_contexts(state=case["state"], task_cids=case["roots"],
                                             output=case["output"], **options)
    assert not case["output"].exists()
    _assert_retained(vectors)


@pytest.mark.parametrize("path", ["admission.json", "prepared.json", "intent.duckdb"])
def test_missing_actual_owner_artifact_refuses_before_context_output(admitted, retained_vectors, path):
    case = admitted
    (case["state"] / path).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _prepare_contexts(case, retained_vectors)
    assert not case["output"].exists()


@pytest.mark.parametrize("mutation", ["source", "event", "blocked", "database", "semantic", "world", "retrieval", "bundle"])
def test_cold_reopen_rechecks_sources_live_owner_and_actual_artifacts(admitted, retained_vectors, mutation):
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    assert _cold(case) == result
    receipt_sha = _hash(case["output"] / "result.json")
    bundle_sha = _hash(case["repository"] / result["context_bundle"]["artifact"])
    if mutation == "source":
        with (case["repository"] / "left.py").open("a") as stream:
            stream.write("\n# changed after context publication\n")
    elif mutation in {"event", "blocked"}:
        with IntentRepository(case["state"] / "intent.duckdb") as intent:
            if mutation == "event":
                intent.upsert_objective(objective_id="cold-reopen-event", objective_alias="COLD", title="Later native event")
            else:
                with intent._connection(write=True) as connection:
                    connection.execute("UPDATE tasks SET status='blocked' WHERE task_cid=?", [case["roots"][0]])
    elif mutation == "database":
        (case["state"] / "intent.duckdb").unlink()
    else:
        metadata = result["contexts"][0]["metadata"]
        artifact = (result["context_bundle"]["artifact"] if mutation == "bundle" else
                    metadata[{"semantic": "Semantic context artifact", "world": "World context artifact",
                              "retrieval": "Code retrieval artifact"}[mutation]])
        (case["repository"] / artifact).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        _cold(case, digest=receipt_sha)
    assert _hash(case["output"] / "result.json") == receipt_sha
    if mutation != "bundle":
        assert _hash(case["repository"] / result["context_bundle"]["artifact"]) == bundle_sha
    if mutation == "database":
        assert not (case["state"] / "intent.duckdb").exists()
    _assert_retained(retained_vectors)


@pytest.mark.parametrize("digest", [None, "", "0" * 64, "f" * 63, "G" * 64])
def test_cold_reopen_requires_the_exact_opaque_expected_digest(admitted, retained_vectors, digest):
    case = admitted
    _prepare_contexts(case, retained_vectors)
    receipt = (case["output"] / "result.json").relative_to(case["repository"]).as_posix()
    with pytest.raises(ValueError):
        contexts.load_ready_task_contexts(state=case["state"], artifact=receipt, expected_sha256=digest)


@pytest.mark.parametrize("field,value", [("index_id", "foreign-index"), ("config_id", "foreign-config"),
    ("dimensions", 768), ("snapshot_sha256", "0" * 64), ("query_id", "foreign-query"),
    ("result_id", "foreign-result"), ("model_revision", "foreign-revision")])
def test_new_outer_digest_cannot_substitute_retained_native_retrieval_identity(admitted, retained_vectors, field, value):
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    result["retrieval_reuse"][field] = value
    digest = _rewrite_receipt(case, result)
    with pytest.raises(ValueError):
        _cold(case, digest=digest)
    _assert_retained(retained_vectors)


@pytest.mark.parametrize("item", [None, [], "foreign-context", True])
def test_cold_reopen_refuses_non_mapping_contexts_with_a_typed_error(admitted, retained_vectors, item):
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    result["contexts"][0] = item
    digest = _rewrite_receipt(case, result)
    with pytest.raises(ValueError):
        _cold(case, digest=digest)


@pytest.mark.parametrize("metadata", [None, [], "foreign-metadata", True])
def test_cold_reopen_refuses_non_mapping_metadata_before_bundle_comparison(admitted, retained_vectors, metadata):
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    result["contexts"][0]["metadata"] = metadata
    digest = _rewrite_receipt(case, result)
    with pytest.raises(ValueError):
        _cold(case, digest=digest)


@pytest.mark.parametrize("mutation", ["program_scope", "retrieval_query", "semantic_query"])
def test_other_valid_producer_context_cannot_replace_signed_task_scope_or_query(admitted, retained_vectors, tmp_path, mutation):
    """Run genuine alternate producers, then require the reviewed owner join."""
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    snapshot, vector_result = retained_vectors["snapshot"], retained_vectors["result"]
    program_paths = case["profile"]["input_paths"]
    paths = case["prepared"]["worker_inputs"]
    if mutation == "program_scope":
        program_paths = ["left.py"]
        paths = [prep.INSTRUCTION, prep.SMOKE, ".supervisor-task-profile.json", "left.py"]
        output = tmp_path / "alternate-retained-vectors"
        qualified = vector_index_preflight.qualify(case["repository"], output, program_paths, case["prepared"]["query"])
        with duckdb.connect(str(output / "vectors.duckdb"), read_only=True, config={"threads": 1}) as connection:
            snapshot = CodeVectorIndexSnapshot.from_dict(json.loads(connection.execute(
                "SELECT payload FROM snapshots WHERE id=?", [qualified["index_id"]]).fetchone()[0]))
        vector_result = CodeVectorSearchResult.from_dict(qualified["hits"])
        result["retrieval_reuse"].update(snapshot_sha256=hashlib.sha256(_wire(snapshot.to_dict())).hexdigest(),
            result_sha256=hashlib.sha256(_wire(vector_result.to_dict())).hexdigest(), index_id=snapshot.index_id,
            config_id=snapshot.config.config_id, query_id=vector_result.query.query_id, result_id=vector_result.result_id,
            dimensions=snapshot.config.dimensions, model_id=snapshot.config.model_id,
            model_revision=snapshot.config.model_revision)
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        alternate = prepare_supervised_task_context(repository=case["repository"], intent=intent,
            task_cid=result["contexts"][0]["task_cid"], paths=paths,
            required_raw_paths=[prep.INSTRUCTION, prep.SMOKE], output=case["repository"] / ".runtime/alternate-context",
            code_vector_snapshot=snapshot, code_vector_result=vector_result,
            code_query_text="Foreign retrieval query" if mutation == "retrieval_query" else case["prepared"]["query"],
            semantic_program_paths=program_paths, semantic_max_symbols=1024,
            semantic_worker_query="Foreign semantic query" if mutation == "semantic_query" else case["prepared"]["query"])
    result["contexts"][0] = alternate
    result["context_bundle"] = write_task_context_bundle(repository=case["repository"], prepared=result["contexts"],
        output=case["repository"] / ".runtime/alternate-context-bundle.json")
    digest = _rewrite_receipt(case, result)
    with pytest.raises(ValueError):
        _cold(case, digest=digest)
    _assert_retained(retained_vectors)


def test_actual_all_task_world_cannot_replace_selected_root_world(admitted, retained_vectors):
    """A valid generic native world still needs the exact reviewed task scope."""
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    item = result["contexts"][0]
    blocks = (case["repository"] / item["metadata"]["Semantic context artifact"]).parent / "blocks"
    output = case["repository"] / ".runtime/all-task-world"
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        capture = capture_intent_world_snapshot(intent, repository_id=item["repository_id"],
            task_cids=sorted(task.task_cid for task in case["tasks"].values()),
            semantic_root_cid=item["semantic_root_cid"], get_semantic_block=lambda cid: (blocks / cid).read_bytes())
        world = persist_intent_world_snapshot(capture, output=output, task_id=item["task_id"])
        native = load_intent_world_context(artifact=output / "intent-world.json", expected_sha256=world["artifact_sha256"],
            task_id=item["task_id"], repository_id=item["repository_id"], intent=intent)
        assert native["intent_freshness_checked"] is True and len(native["tasks"]) == 3
    item["world"] = world
    item["world_snapshot_cid"] = capture["snapshot"]["snapshot_cid"]
    item["plan_projection_cid"] = capture["plan_projection"]["projection_cid"]
    item["metadata"].update({"World context artifact": (output / "intent-world.json").relative_to(case["repository"]).as_posix(),
                             "World context sha256": world["artifact_sha256"]})
    result["context_bundle"] = write_task_context_bundle(repository=case["repository"], prepared=result["contexts"],
        output=case["repository"] / ".runtime/all-task-world-bundle.json")
    digest = _rewrite_receipt(case, result)
    with pytest.raises(ValueError, match="world|semantic"):
        _cold(case, digest=digest)


@pytest.mark.parametrize("variant", ["inline", "reference", "downgrade"])
def test_rehashed_source384_bundle_cannot_change_plain_context_route(admitted, retained_vectors, tmp_path, variant):
    """Authored transport declarations grant no model validation or inference."""
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    bundle_path = case["repository"] / result["context_bundle"]["artifact"]
    payload = json.loads(bundle_path.read_bytes())
    if variant != "downgrade":
        payload["schema"] = bundles.SOURCE384_REFERENCE_SCHEMA if variant == "reference" else bundles.SOURCE384_SCHEMA
    for task in payload["tasks"]:
        task["source384_context"] = ({"schema": bundles.RECEIPT_REFERENCE_SCHEMA,
            "output": str(tmp_path / "unselected-source384-context"), "receipt_sha256": "0" * 64,
            "receipt_bytes": 1} if variant == "reference" else
            {"schema": "terminal-source384-repository-context@1", "completion_authority": False})
    bundle_path.write_bytes(_wire(payload))
    result["context_bundle"]["sha256"] = _hash(bundle_path)
    digest = _rewrite_receipt(case, result)
    with pytest.raises(ValueError, match="plain native nomination"):
        _cold(case, digest=digest)
    assert not (tmp_path / "unselected-source384-context").exists()
    _assert_retained(retained_vectors)


def test_cold_reopen_refuses_selected_catalog_generation_edit_without_loading_weights(admitted, retained_vectors, tmp_path):
    from test.test_ir_persistent_catalog import _record, _selection, _raw
    case = admitted
    record = _record("codebase_ir", 384, "input_embedding")
    catalog = tmp_path / "selected-ir-model-manager-catalog.json"
    catalog.write_bytes(_raw([record]))
    selections = {cid: [_selection(record)] for cid in case["roots"]}
    result = contexts.prepare_ready_task_contexts(state=case["state"], task_cids=case["roots"], output=case["output"],
        code_vector_snapshot=retained_vectors["snapshot"], code_vector_result=retained_vectors["result"],
        ir_catalog_path=catalog, ir_selections=selections)
    assert _cold(case) == result
    assert result["ir_selection_mode"] == "metadata_nomination"
    assert result["checkpoint_bytes_authenticated"] is result["decoder_runtime_admitted"] is False
    receipt_sha = _hash(case["output"] / "result.json")
    bundle_sha = _hash(case["repository"] / result["context_bundle"]["artifact"])
    record["huggingface_config"]["ir_checkpoint"]["trained"] = True
    catalog.write_bytes(_raw([record]))
    with pytest.raises(ValueError, match="catalog|selection"):
        _cold(case, digest=receipt_sha)
    assert _hash(case["output"] / "result.json") == receipt_sha
    assert _hash(case["repository"] / result["context_bundle"]["artifact"]) == bundle_sha


def test_cold_reopen_rejects_an_oversized_otherwise_unchanged_native_admission(admitted, retained_vectors):
    case = admitted
    _prepare_contexts(case, retained_vectors)
    receipt_sha = _hash(case["output"] / "result.json")
    admission = case["state"] / "admission.json"
    admission.write_bytes(b" " * (4 * 1024 * 1024 + 1) + admission.read_bytes())
    with pytest.raises(ValueError, match="bounded|admission"):
        _cold(case, digest=receipt_sha)


def test_actual_active_block_with_ready_status_still_refuses_context(admitted, retained_vectors):
    case = admitted
    selected = case["roots"][0]
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        intent.block_task(task_cid=selected, blocker_kind="review", blocker_id="context-review",
                          reason="Retained active blocker")
        # Corrupt only status back to ready; the actual active blocker remains.
        with intent._connection(write=True) as connection:
            connection.execute("UPDATE tasks SET status='ready' WHERE task_cid=?", [selected])
        assert intent.get_task(selected)["status"] == "ready"
        assert not intent.select_ready_tasks(task_cids=[selected])
    with pytest.raises(ValueError, match="ready"):
        _prepare_contexts(case, retained_vectors, selected=[selected])
    assert not case["output"].exists()


def test_two_roots_with_actual_empty_program_population_need_no_vectors(multitask_case):
    case = multitask_case
    for path in case["profile"]["input_paths"]:
        (case["repository"] / path).write_text("# exact signed empty program population\n")
    prepared = _prepare(case)
    assert prep.plan(case["state"])["qualified"]
    admission = json.loads((case["state"] / "admission.json").read_bytes())
    tasks = PromptGoalGraph.from_dict(admission["graph"]).tasks
    roots = sorted(task.task_cid for task in tasks if not task.dependency_task_cids)
    case = {**case, "prepared": prepared, "admission": admission, "roots": roots,
            "output": case["repository"] / ".runtime/empty-ready-task-contexts"}
    result = _prepare_contexts(case)
    assert result["task_cids"] == roots and len(result["contexts"]) == 2
    assert result["retrieval_reuse"] == {"status": "verified_empty_program", "index_id": None}
    assert _cold(case) == result
    assert all(result[name] == 0 for name in ("new_embedding_calls", "model_loading_calls", "training_steps", "provider_calls"))
    assert result["execution_authority"] is result["completion_authority"] is result["proof_authority"] is False
    for operation in (prep.initial_context, prep.context):
        with pytest.raises(ValueError, match="administrative"):
            operation(state=case["state"])
    assert not (case["state"] / "initial-context-result.json").exists()


def test_controlled_status_change_after_actual_ready_selection_refuses_before_outputs(admitted, retained_vectors, monkeypatch):
    """Preserve actual readiness, then corrupt the live selected task status."""
    case = admitted
    selected_actual = IntentRepository.select_ready_tasks
    events = []

    def block_after_actual_ready(intent, *args, **kwargs):
        selected = selected_actual(intent, *args, **kwargs)
        if not events:
            with intent._connection(write=True) as connection:
                connection.execute("UPDATE tasks SET status='blocked' WHERE task_cid=?", [case["roots"][0]])
            events.append(True)
        return selected

    monkeypatch.setattr(IntentRepository, "select_ready_tasks", block_after_actual_ready)
    with pytest.raises(ValueError, match="root|ready"):
        _prepare_contexts(case, retained_vectors)
    assert events == [True]
    assert not case["output"].exists()


def test_controlled_source_edit_after_actual_task_context_refuses_bundle_publication(admitted, retained_vectors, monkeypatch):
    """Retain a genuine context result, then mutate its actual signed source."""
    case = admitted
    prepare_actual = contexts.prepare_supervised_task_context
    events = []

    def change_source_after_context(*args, **kwargs):
        context = prepare_actual(*args, **kwargs)
        with (case["repository"] / "left.py").open("a") as stream:
            stream.write("\n# changed after actual context production\n")
        events.append(True)
        return context

    monkeypatch.setattr(contexts, "prepare_supervised_task_context", change_source_after_context)
    with pytest.raises(ValueError, match="stale|changed|source"):
        _prepare_contexts(case, retained_vectors)
    assert events == [True]
    assert not (case["output"] / "task-context-bundle.json").exists()
    assert not (case["output"] / "result.json").exists()


def test_nomination_identity_and_current_world_are_checked_independently(admitted, retained_vectors):
    case = admitted
    result = _prepare_contexts(case, retained_vectors)
    left, right = sorted(result["contexts"], key=lambda row: row["task_id"])
    bundle = result["context_bundle"]
    with pytest.raises(ValueError, match="foreign task"):
        load_task_context_nomination(repository=case["repository"], artifact=bundle["artifact"],
            expected_sha256=bundle["sha256"], task_cid=left["task_cid"], task_id=right["task_id"])
    with IntentRepository(case["state"] / "intent.duckdb") as intent:
        intent.upsert_objective(objective_id="later-context-event", objective_alias="LATER", title="Later observation")
        for item in result["contexts"]:
            metadata = item["metadata"]
            with pytest.raises(ValueError, match="stale"):
                load_intent_world_context(artifact=case["repository"] / metadata["World context artifact"],
                    expected_sha256=metadata["World context sha256"], task_id=item["task_id"],
                    repository_id=metadata["World context repository"], intent=intent)


def test_controlled_native_event_after_actual_world_capture_prevents_bundle_publication(admitted, retained_vectors, monkeypatch):
    """Preserve the real capture, then inject an actual owner event."""
    from ipfs_accelerate_py.agent_supervisor.runtime import supervised_task_context
    case = admitted
    capture_actual = supervised_task_context.capture_intent_world_snapshot
    events = []

    def change_owner_after_capture(intent, *args, **kwargs):
        captured = capture_actual(intent, *args, **kwargs)
        if not events:
            intent.upsert_objective(objective_id="during-world-capture", objective_alias="DURING", title="Concurrent owner event")
            events.append(True)
        return captured

    monkeypatch.setattr(supervised_task_context, "capture_intent_world_snapshot", change_owner_after_capture)
    with pytest.raises(ValueError, match="changed|stale"):
        _prepare_contexts(case, retained_vectors)
    assert events == [True]
    assert not list(case["output"].rglob("*bundle*.json"))
