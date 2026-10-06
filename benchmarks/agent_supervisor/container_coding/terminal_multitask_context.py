"""Observe ready root-task contexts without enabling multi-task execution.

This explicit administrative route reuses supplied native retrieval objects.
It never builds embeddings, trains a decoder, claims tasks or grants START.
Existing singleton context and reviewed-profile execution gates stay in place.
Currentness checks are cooperative observations, not an atomic filesystem or
native database snapshot.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import stat

from ipfs_accelerate_py.agent_supervisor.analysis.code_symbol_vector_index import (
    CodeVectorIndexSnapshot, CodeVectorSearchResult,
)
from ipfs_accelerate_py.agent_supervisor.runtime import code_retrieval_context as retrieval
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime import task_context_bundle as bundles
from ipfs_accelerate_py.agent_supervisor.runtime.empty_code_retrieval import observe_empty_program_population
from ipfs_accelerate_py.agent_supervisor.runtime.semantic_context_runtime import load_semantic_worker_context
from ipfs_accelerate_py.agent_supervisor.runtime.supervised_task_context import prepare_supervised_task_context
from ipfs_accelerate_py.agent_supervisor.runtime.task_context_bundle import (
    load_task_context_selection, write_task_context_bundle,
)
from ipfs_accelerate_py.agent_supervisor.runtime.terminal_task_profile import (
    INSTRUCTION, MULTITASK_SCHEMA, PROFILE, SMOKE,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.intent_world_snapshot import load_intent_world_context
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository

from . import terminal_indexed_preparation as preparation


SCHEMA = "terminal-reviewed-ready-task-contexts@1"
CHECKPOINT_SCHEMA = "terminal-reviewed-ready-task-contexts@2"
MAX_TASKS = 16
MAX_SOURCE_BYTES = 1_000_000
MAX_INPUT_FILES = 64
MAX_QUERY_BYTES = 8192
MAX_RESULT_BYTES = 8_000_000


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _freeze(value):
    return json.loads(_wire(value))


def _digest(value):
    return hashlib.sha256(_wire(value)).hexdigest()


def _relations(spec):
    return {
        "outputs": [{"ordinal": index, "path": row["path"], "effect": row}
                    for index, row in enumerate(spec["outputs"])],
        "acceptance": [{"ordinal": index, "criterion": row["criterion"], "evidence_policy": row}
                       for index, row in enumerate(spec["acceptance"])],
        "validations": [{"ordinal": index, "argv": row["argv"],
                         "policy": {name: value for name, value in row.items() if name != "argv"}}
                        for index, row in enumerate(spec["validations"])],
    }


def _database_identity(path):
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("ready task context requires the exact existing file-backed intent owner")
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("ready task context requires the exact existing file-backed intent owner")
    return {"path": str(path), "device": info.st_dev, "inode": info.st_ino}


def _read_regular(path, limit):
    if path.is_symlink() or path.resolve(strict=True) != path:
        raise ValueError("context artifact must be an exact regular file")
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_size > limit:
            raise ValueError("context artifact must be a bounded regular file")
        raw = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    witness = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if (witness(before) != witness(after) or witness(path.lstat()) != witness(before)
            or path.resolve(strict=True) != path or len(raw) > limit or len(raw) != before.st_size):
        raise ValueError("context artifact changed during its bounded native read")
    return raw


def _admission(state):
    raw = _read_regular(state / "admission.json", 4 * 1024 * 1024)
    return json.loads(raw, object_pairs_hook=retrieval_unique), hashlib.sha256(raw).hexdigest()


def _partition(profile, manifest):
    return {"program_paths": profile["input_paths"], "support_hashes": {
        name: {"role": role, "sha256": manifest["sources"][name]["sha256"]}
        for name, role in ((INSTRUCTION, "instruction"), (SMOKE, "structural_smoke"), (PROFILE, "task_profile"))}}


def _source_inputs(repository, prepared):
    from ipfs_accelerate_py.agent_supervisor.runtime.terminal_source_partition import _read
    paths = prepared["worker_inputs"]
    if not 1 <= len(paths) <= MAX_INPUT_FILES:
        raise ValueError("ready task context source population exceeds the native 64-file bound")
    total = 0
    for name in paths:
        raw = _read(repository, name, prepared["manifest"]["payload"]["sources"], MAX_SOURCE_BYTES)
        raw.decode("utf-8")
        total += len(raw)
        if len(raw) > MAX_SOURCE_BYTES or total > MAX_SOURCE_BYTES:
            raise ValueError("ready task context source population exceeds the native byte bound")
    query = prepared["query"]
    if not query.strip() or len(query.encode("utf-8")) > MAX_QUERY_BYTES:
        raise ValueError("ready task context query exceeds the native 8192-byte bound")


def _native_observation(intent, admission, verified, selected):
    """Read exact signed tasks and native dependency readiness without mutation."""
    before = intent.event_watermark()
    projection = local._plain(dict(intent.plan_projection()))
    expected = {task.task_cid: task for task in verified["graph"].tasks}
    if {row["task_cid"] for row in projection["tasks"]} != set(expected):
        raise ValueError("native task population differs from the complete admitted graph")
    ready = {row["task_cid"]: dict(row) for row in intent.select_ready_tasks(
        limit=MAX_TASKS + 1, task_cids=selected, automatic_only=False,
    )}
    if set(ready) != set(selected):
        raise ValueError("selected task is not ready under the native dependency, block and cooldown gates")
    rows = []
    objective = local.content_identity({"manifest": verified["receipt"]["manifest_cid"],
                                       "objective": verified["graph"].root_goal.objective})
    for cid, task in sorted(expected.items()):
        native = intent.get_task(cid)
        if native is None:
            raise ValueError("native task disappeared during context observation")
        row = local._plain(dict(native))
        envelope = row["body"].get(local.CONTRACT_KEY, {})
        contract = local._verify_signature(envelope, verified["profile"])
        wanted = local._pending_contract_payload(admission=admission, verified=verified,
            task=task, intent_owner_id=intent.owner_id)
        spec = wanted["task_spec"]
        identity = {"local_contract_cid": local.content_identity(envelope),
                    "repository_tree_id": verified["receipt"]["source_tree_id"],
                    "task_cid": cid, "task_alias": task.task_key}
        if (contract != wanted or set(row["body"]) != {"title", local.CONTRACT_KEY}
                or row["identity"] != identity or row["task_alias"] != task.task_key
                or row["goal_cid"] != task.goal_cid or row["plan_cid"] != wanted["plan_id"]
                or row["objective_id"] != objective or row["dependencies"] != wanted["dependencies"]
                or row["body"].get("title") != task.objective
                or {name: row[name] for name in ("outputs", "acceptance", "validations")} != _relations(spec)):
            raise ValueError("native task differs from its complete signed pending contract and relations")
        plan = intent.get_plan(row["plan_cid"])
        retained = local.load_local_planning_receipt(
            plan["body"].get("local_planning_receipt_ref", {}) if plan else {},
            manifest=admission["manifest"],
        )
        if retained != admission["receipt"]:
            raise ValueError("native plan differs from the retained complete planning receipt")
        if cid in selected and (row["status"] != "ready" or row["dependencies"]):
            raise ValueError("ready task context currently requires independent root tasks; predecessor source binding is unqualified")
        rows.append(row)
    if intent.event_watermark() != before:
        raise ValueError("native task state changed during context observation")
    return {"event_watermark": before, "plan_projection_cid": projection["projection_cid"],
            "native_rows_sha256": _digest(rows), "selected_ready": local._plain(ready),
            "tasks": [{"task_cid": row["task_cid"], "task_id": row["task_alias"],
                       "revision": row["revision"], "status": row["status"],
                       "dependencies": row["dependencies"],
                       "local_contract_cid": row["identity"]["local_contract_cid"]}
                      for row in rows if row["task_cid"] in selected]}


def _resolve_ir(catalog_path, selections, selected):
    if catalog_path is None:
        return None
    from ipfs_accelerate_py.agent_supervisor.runtime.task_ir_selection import resolve_task_ir_selections
    results = {cid: resolve_task_ir_selections(catalog_path=catalog_path, selections=selections[cid])
               for cid in selected}
    generations = {(record["binding_snapshot_revision"], record["catalog_revision"])
                   for records in results.values() for record in records}
    if (len(generations) != 1 or resolve_task_ir_selections(
            catalog_path=catalog_path, selections=selections[selected[0]]) != results[selected[0]]):
        raise ValueError("per-task IR selections do not share one unchanged logical catalog generation")
    return results


def _authenticate_ir(catalog_path, selections, selected, metadata):
    """Observe original checkpoint bytes without promoting a decoder runtime."""
    from ipfs_accelerate_py.agent_supervisor.runtime.task_ir_checkpoint import (
        MAX_TOTAL_CHECKPOINT_BYTES, authenticate_task_ir_checkpoints,
    )
    total = sum(row["selected_binding"]["declaration"]["original_checkpoint_pin"]["bytes"]
                for cid in selected for row in metadata[cid])
    if total > MAX_TOTAL_CHECKPOINT_BYTES:
        raise ValueError("per-task checkpoint population exceeds the aggregate byte bound")
    observed = {cid: authenticate_task_ir_checkpoints(catalog_path=catalog_path,
        selections=selections[cid]) for cid in selected}
    if any([row["native_resolution"] for row in observed[cid]] != metadata[cid]
           for cid in selected):
        raise ValueError("checkpoint observations differ from exact per-task catalog nominations")
    # Close the per-task batch against a fresh complete metadata generation.
    if _resolve_ir(catalog_path, selections, selected) != metadata:
        raise ValueError("checkpoint catalog changed during per-task authentication")
    _checkpoint_file_fence(observed)
    return observed


def _checkpoint_file_fence(observations):
    """Close all earlier task files after later cooperative owner observations."""
    for rows in observations.values():
        for row in rows:
            pin, witness = row["original_checkpoint_pin"], row["file_witness"]
            path = Path(pin["path"])
            if path.resolve(strict=True) != path:
                raise ValueError("checkpoint path changed after task authentication")
            info = path.lstat()
            current = {"path": str(path), "bytes": info.st_size, "sha256": pin["sha256"],
                "device": info.st_dev, "inode": info.st_ino, "mode": info.st_mode,
                "nlink": info.st_nlink, "mtime_ns": info.st_mtime_ns, "ctime_ns": info.st_ctime_ns}
            if (not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or current != witness):
                raise ValueError("checkpoint file identity changed after task authentication")


def _verify_plain_bundle(repository, reference, selected):
    path = retrieval._path(repository, reference["artifact"])
    if not path.is_relative_to(repository / ".runtime"):
        raise ValueError("ready task bundle requires the separate repository runtime namespace")
    raw = _read_regular(path, bundles.MAX_BYTES)
    if hashlib.sha256(raw).hexdigest() != reference["sha256"]:
        raise ValueError("ready task bundle differs from its exact nomination digest")
    payload = json.loads(raw, object_pairs_hook=retrieval_unique)
    if (type(payload) is not dict
            or set(payload) != {"schema", "tasks", "completion_authority"}
            or payload["schema"] != bundles.SCHEMA
            or payload["completion_authority"] is not False
            or type(payload["tasks"]) is not list
            or not 1 <= len(payload["tasks"]) <= MAX_TASKS
            or any(type(item) is not dict or set(item) != {"task_cid", "task_id", "metadata"}
                   for item in payload["tasks"])
            or [item["task_cid"] for item in payload["tasks"]] != list(selected)
            or bundles._raw(payload) != raw):
        raise ValueError("ready task bundle must be the bounded plain native nomination")


def _verify_context(repository, intent, context, prepared, profile):
    fields = {"schema", "task_cid", "task_id", "task_title", "task_revision", "plan_projection_cid",
              "event_watermark", "repository_id", "semantic_root_cid", "world_snapshot_cid", "metadata",
              "semantic", "world", "retrieval", "execution_authority", "completion_authority", "canonical_task_mutated"}
    if (type(context) is not dict or set(context) != fields
            or context["schema"] != "supervisor-task-context-preparation@1"
            or any(context[name] is not False for name in (
                "execution_authority", "completion_authority", "canonical_task_mutated"))):
        raise ValueError("ready task context differs from the native advisory producer")
    metadata = context["metadata"]
    keys = {"Semantic context artifact", "Semantic context sha256", "Semantic context refresh",
            "World context artifact", "World context sha256", "World context repository",
            "Code retrieval artifact", "Code retrieval sha256"}
    if (type(metadata) is not dict or set(metadata) != keys
            or any(type(value) is not str for value in metadata.values())
            or metadata["Semantic context refresh"] != "true"
            or any(type(context[name]) is not dict for name in ("semantic", "world", "retrieval"))):
        raise ValueError("ready task context metadata differs from the closed native producer")
    for kind in ("Semantic context", "World context", "Code retrieval"):
        path = retrieval._path(repository, metadata[kind + " artifact"])
        if not path.is_relative_to(repository / ".runtime") or path.resolve(strict=True) != path:
            raise ValueError("ready task context requires exact repository runtime artifact nominations")
    world_path = retrieval._path(repository, metadata["World context artifact"])
    world_raw = _read_regular(world_path, MAX_RESULT_BYTES)
    if hashlib.sha256(world_raw).hexdigest() != metadata["World context sha256"]:
        raise ValueError("ready task world differs from its bounded canonical artifact")
    semantic = json.loads(load_semantic_worker_context(repository=repository,
        artifact=metadata["Semantic context artifact"],
        expected_sha256=metadata["Semantic context sha256"], task_id=context["task_id"]))
    world = load_intent_world_context(artifact=world_path,
        expected_sha256=metadata["World context sha256"], task_id=context["task_id"],
        repository_id=metadata["World context repository"], intent=intent)
    task_projection = local._plain(dict(intent.plan_projection(task_cids=[context["task_cid"]])))
    if (semantic["semantic_root_cid"] != context["semantic_root_cid"]
            or {name: row["sha256"] for name, row in semantic["manifest"].items()} != {
                name: row["sha256"] for name, row in prepared["manifest"]["payload"]["sources"].items()}
            or semantic.get("program_paths") != profile["input_paths"]
            or semantic["required_raw_paths"] != sorted([INSTRUCTION, SMOKE])
            or semantic.get("worker_projection", {}).get("query") != prepared["query"]
            or semantic["objective"] != context["task_title"]
            or context["semantic"].get("task_id") != context["task_id"]
            or context["semantic"].get("semantic_root_cid") != context["semantic_root_cid"]
            or context["semantic"].get("scope_cid") != semantic["scope_cid"]
            or context["semantic"].get("worker_payload_sha256") != metadata["Semantic context sha256"]
            or context["semantic"].get("worker_capsules") != len(semantic["capsules"])
            or context["semantic"].get("completion_authority") is not False
            or context["world"].get("planning_context") != {name: value for name, value in world.items()
                                                            if name != "intent_freshness_checked"}
            or context["world"].get("artifact_sha256") != metadata["World context sha256"]
            or context["world"].get("execution_authority") is not False
            or context["world"].get("completion_authority") is not False
            or context["repository_id"] != world["repository_id"]
            or world["semantic_root_cid"] != context["semantic_root_cid"]
            or world["plan_projection_cid"] != task_projection["projection_cid"]
            or world["tasks"] != task_projection["tasks"]
            or world["world_snapshot_cid"] != context["world_snapshot_cid"]
            or world["plan_projection_cid"] != context["plan_projection_cid"]
            or world["event_watermark"] != context["event_watermark"]):
        raise ValueError("task semantic and live world identities differ")
    code = json.loads(retrieval.load_code_retrieval_context(repository=repository,
        artifact=metadata["Code retrieval artifact"],
        expected_sha256=metadata["Code retrieval sha256"], task_id=context["task_id"]))
    if code["status"] != "current":
        raise ValueError("ready task retrieval source is stale")
    program_hashes = {name: prepared["manifest"]["payload"]["sources"][name]["sha256"]
                      for name in profile["input_paths"]}
    if code["query_text"] != prepared["query"] or code["source_sha256"] != program_hashes:
        raise ValueError("ready task retrieval query or complete signed source scope differs")
    if code.get("retrieval_schema") == "supervisor-empty-code-retrieval@1":
        partition = _partition(profile, prepared["manifest"]["payload"])
        if code["program_paths"] != partition["program_paths"] or code["support_hashes"] != partition["support_hashes"]:
            raise ValueError("ready task empty retrieval differs from the complete signed support partition")
    observed = context["retrieval"]
    if (observed.get("sha256") != metadata["Code retrieval sha256"]
            or any(observed.get(name) != code[name] for name in ("index_id", "query_id", "result_id"))
            or observed.get("execution_authority") is not False or observed.get("completion_authority") is not False):
        raise ValueError("ready task retrieval observations differ from the live native artifact")


def _retrieval_binding(repository, context):
    metadata = context["metadata"]
    payload = json.loads(retrieval._read(retrieval._path(repository, metadata["Code retrieval artifact"]),
                                        retrieval.MAX_ARTIFACT_BYTES), object_pairs_hook=retrieval_unique)
    if payload["schema"] == "supervisor-empty-code-retrieval@1":
        return {"status": "verified_empty_program", "index_id": None}
    if "snapshot_ref" in payload:
        reference = payload["snapshot_ref"]
        snapshot_payload = json.loads(retrieval._read(retrieval._path(repository, reference["path"]),
                                                      reference["bytes"]), object_pairs_hook=retrieval_unique)
    else:
        snapshot_payload = payload["snapshot"]
    snapshot = CodeVectorIndexSnapshot.from_dict(snapshot_payload)
    result = CodeVectorSearchResult.from_dict(payload["result"])
    return {"status": "existing_native_objects", "snapshot_sha256": _digest(snapshot.to_dict()),
            "result_sha256": _digest(result.to_dict()), "index_id": snapshot.index_id,
            "config_id": snapshot.config.config_id, "query_id": result.query.query_id,
            "result_id": result.result_id, "model_id": snapshot.config.model_id,
            "model_revision": snapshot.config.model_revision, "dimensions": snapshot.config.dimensions}


def prepare_ready_task_contexts(*, state: Path, task_cids: Sequence[str], output: Path,
        code_vector_snapshot=None, code_vector_result=None,
        ir_catalog_path: Path | None = None, ir_selections: Mapping[str, list[dict]] | None = None,
        authenticate_ir_checkpoints: bool = False) -> dict:
    """Prepare advisory contexts for explicitly selected admitted ready roots.

    Nonempty source programs require existing complete native retrieval objects.
    This function replays their numerical/source bindings, without proving that
    a supplied query vector represents the instruction's semantics. Optional IR
    bindings nominate metadata only; checkpoint bytes and runtime usability are
    independent future gates. Explicit checkpoint authentication additionally
    streams original registered byte pins and binds file identities; it does
    not deserialize weights, verify an ABI or admit a runtime.
    """
    if type(authenticate_ir_checkpoints) is not bool:
        raise ValueError("checkpoint authentication requires an explicit boolean selection")
    if authenticate_ir_checkpoints and (ir_catalog_path is None or ir_selections is None):
        raise ValueError("checkpoint authentication requires exact per-task IR catalog selections")
    if (isinstance(task_cids, (str, bytes)) or not isinstance(task_cids, Sequence)
            or not 1 <= len(task_cids) <= MAX_TASKS
            or any(type(cid) is not str or not cid for cid in task_cids)
            or len(set(task_cids)) != len(task_cids)):
        raise ValueError("ready task context requires 1 to 16 unique explicit admitted task CIDs")
    selected = tuple(sorted(task_cids))
    state = Path(state).absolute()
    if state.resolve(strict=True) != state or not state.is_dir():
        raise ValueError("ready task context state must be its exact existing canonical directory")
    prepared = preparation._load_prepared(state)
    if not preparation._is_multitask(prepared):
        raise ValueError("ready task context requires the explicit reviewed multi-task profile")
    repository = Path(prepared["repository"])
    admission, admission_identity = _admission(state)
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    if admission["manifest"] != prepared["manifest"]:
        raise ValueError("ready task context admission differs from the independently prepared manifest")
    expected = {task.task_cid: task for task in verified["graph"].tasks}
    if not set(selected) <= set(expected):
        raise ValueError("selected ready task CID is absent from the admitted graph")
    if any(expected[cid].dependency_task_cids for cid in selected):
        raise ValueError("ready task context currently requires independent root tasks; predecessor source binding is unqualified")
    profile = local._verify_multitask_profile_sources(verified["manifest"])
    if profile is None or profile["schema"] != MULTITASK_SCHEMA:
        raise ValueError("ready task context profile differs from the exact signed declaration")
    _source_inputs(repository, prepared)
    output = Path(output).absolute()
    runtime = repository / ".runtime"
    if (output.resolve() != output or output.exists() or output == runtime
            or not output.is_relative_to(runtime)
            or any((repository / name).is_relative_to(output) for name in prepared["worker_inputs"])):
        raise ValueError("ready task contexts require a new separate canonical repository runtime subtree")
    database = state / "intent.duckdb"
    database_identity = _database_identity(database)
    if (ir_catalog_path is None) != (ir_selections is None):
        raise ValueError("IR metadata requires both explicit catalog and per-task selections")
    if ir_selections is not None:
        if not isinstance(ir_selections, Mapping) or set(ir_selections) != set(selected):
            raise ValueError("IR metadata selection must bind exactly the selected task CIDs")
        ir_selections = _freeze(dict(ir_selections))
    ir_metadata = _resolve_ir(ir_catalog_path, ir_selections, selected)
    ir_checkpoints = (_authenticate_ir(ir_catalog_path, ir_selections, selected, ir_metadata)
                      if authenticate_ir_checkpoints else None)
    program_paths = profile["input_paths"]
    partition = _partition(profile, verified["manifest"])
    if (code_vector_snapshot is None) != (code_vector_result is None):
        raise ValueError("existing native retrieval requires both snapshot and result")
    retrieval_identity = None
    if code_vector_snapshot is not None:
        if type(code_vector_snapshot) is not CodeVectorIndexSnapshot or type(code_vector_result) is not CodeVectorSearchResult:
            raise TypeError("ready task context requires actual native retrieval snapshot and result objects")
        if set(code_vector_snapshot.included_paths) != set(program_paths):
            raise ValueError("reused retrieval scope differs from all exact signed program inputs")
        retrieval._replay(code_vector_snapshot, code_vector_result)
        sources, hashes = retrieval._sources(repository, code_vector_snapshot.included_paths)
        retrieval._verified_sources(code_vector_snapshot, sources)
        if hashes != {name: verified["manifest"]["sources"][name]["sha256"] for name in program_paths}:
            raise ValueError("reused retrieval differs from exact signed source bytes")
        retrieval_identity = {"snapshot_sha256": _digest(code_vector_snapshot.to_dict()),
                              "result_sha256": _digest(code_vector_result.to_dict())}
    else:
        if observe_empty_program_population(repository=repository, **partition) is None:
            raise ValueError("nonempty program context requires supplied existing native retrieval objects")
    with IntentRepository(database, install_schema=False) as intent:
        native = _native_observation(intent, admission, verified, selected)

        def current():
            if (preparation._load_prepared(state) != prepared
                    or _admission(state)[1] != admission_identity
                    or local.verify_local_benchmark_admission(admission, initial=True)["manifest"] != verified["manifest"]
                    or _database_identity(database) != database_identity
                    or _native_observation(intent, admission, verified, selected) != native
                    or _resolve_ir(ir_catalog_path, ir_selections, selected) != ir_metadata
                    or (authenticate_ir_checkpoints and _authenticate_ir(
                        ir_catalog_path, ir_selections, selected, ir_metadata) != ir_checkpoints)):
                raise ValueError("ready task context profile, source, task owner or IR selection changed")
            if authenticate_ir_checkpoints:
                # Checkpoint reads may be long. Close source/native owners
                # after them, then close every earlier checkpoint witness.
                if (preparation._load_prepared(state) != prepared
                        or _admission(state)[1] != admission_identity
                        or _database_identity(database) != database_identity
                        or _native_observation(intent, admission, verified, selected) != native
                        or _resolve_ir(ir_catalog_path, ir_selections, selected) != ir_metadata):
                    raise ValueError("ready task context owners changed during checkpoint authentication")
                _checkpoint_file_fence(ir_checkpoints)
            if retrieval_identity is not None and retrieval_identity != {
                    "snapshot_sha256": _digest(code_vector_snapshot.to_dict()),
                    "result_sha256": _digest(code_vector_result.to_dict())}:
                raise ValueError("selected native retrieval objects changed during reuse")

        current()
        output.mkdir(parents=True, mode=0o700, exist_ok=False)
        output.chmod(0o700)
        contexts = []
        for cid in selected:
            context = prepare_supervised_task_context(repository=repository, intent=intent,
                task_cid=cid, paths=prepared["worker_inputs"], required_raw_paths=[INSTRUCTION, SMOKE],
                output=output / hashlib.sha256(cid.encode("utf-8")).hexdigest(),
                code_vector_snapshot=code_vector_snapshot, code_vector_result=code_vector_result,
                code_empty_population=partition if retrieval_identity is None else None,
                code_query_text=prepared["query"], semantic_program_paths=program_paths,
                semantic_max_symbols=1024, semantic_worker_query=prepared["query"], semantic_worker_max_bytes=32768)
            _verify_context(repository, intent, context, prepared, profile)
            current()
            contexts.append(context)
        current()
        for context in contexts:
            _verify_context(repository, intent, context, prepared, profile)
            expected_retrieval = ({"status": "existing_native_objects", **retrieval_identity,
                "index_id": code_vector_snapshot.index_id, "config_id": code_vector_snapshot.config.config_id,
                "query_id": code_vector_result.query.query_id, "result_id": code_vector_result.result_id,
                "model_id": code_vector_snapshot.config.model_id,
                "model_revision": code_vector_snapshot.config.model_revision,
                "dimensions": code_vector_snapshot.config.dimensions} if retrieval_identity else
                {"status": "verified_empty_program", "index_id": None})
            if _retrieval_binding(repository, context) != expected_retrieval:
                raise ValueError("prepared context differs from the exact selected native retrieval objects")
        current()
        bundle = write_task_context_bundle(repository=repository, prepared=contexts,
                                           output=output / "task-context-bundle.json")
        _verify_plain_bundle(repository, bundle, selected)
        for context in contexts:
            selected_context = load_task_context_selection(repository=repository,
                artifact=bundle["artifact"], expected_sha256=bundle["sha256"],
                task_cid=context["task_cid"], task_id=context["task_id"])
            if selected_context["metadata"] != {name.lower(): value for name, value in context["metadata"].items()}:
                raise ValueError("ready task bundle differs from the verified native context")
        current()
        result = {
            "schema": CHECKPOINT_SCHEMA if authenticate_ir_checkpoints else SCHEMA,
            "repository": str(repository), "state": str(state),
            "task_cids": list(selected), "contexts": contexts, "context_bundle": bundle,
            "manifest_cid": local.content_identity(admission["manifest"]),
            "admission_cid": local.content_identity(admission),
            "requirement_contract_cid": profile["intent_requirement_contract_cid"],
            "native_binding": {"database": database_identity, **native},
            "source_hashes": {name: row["sha256"] for name, row in verified["manifest"]["sources"].items()},
            "retrieval_reuse": ({"status": "existing_native_objects", **retrieval_identity,
                "index_id": code_vector_snapshot.index_id, "config_id": code_vector_snapshot.config.config_id,
                "query_id": code_vector_result.query.query_id, "result_id": code_vector_result.result_id,
                "model_id": code_vector_snapshot.config.model_id,
                "model_revision": code_vector_snapshot.config.model_revision,
                "dimensions": code_vector_snapshot.config.dimensions} if retrieval_identity else
                {"status": "verified_empty_program", "index_id": None}),
            "ir_selection_mode": ("authenticated_checkpoint_nomination" if authenticate_ir_checkpoints else
                "metadata_nomination" if ir_metadata is not None else "structural_source_context"),
            "ir_catalog_path": str(ir_catalog_path) if ir_metadata is not None else None,
            "ir_selections": ir_selections,
            "ir_metadata_nominations": ir_metadata,
            "checkpoint_bytes_authenticated": authenticate_ir_checkpoints, "decoder_runtime_admitted": False,
            "new_embedding_calls": 0, "model_loading_calls": 0, "training_steps": 0, "provider_calls": 0,
            "query_semantic_alignment_verified": False, "proof_authority": False,
            "execution_authority": False, "completion_authority": False, "canonical_task_mutated": False,
        }
        if authenticate_ir_checkpoints:
            result["ir_checkpoint_observations"] = ir_checkpoints
        raw = _wire(result)
        if len(raw) > MAX_RESULT_BYTES:
            raise ValueError("ready task context receipt exceeds its byte bound")
        with (output / "result.json").open("xb") as stream:
            stream.write(raw)
    return result


def load_ready_task_contexts(*, state: Path, artifact: str, expected_sha256: str,
        require_checkpoint_authentication: bool = False) -> dict:
    """Cold-reopen a sealed nomination against current source and native owners."""
    if type(require_checkpoint_authentication) is not bool:
        raise ValueError("checkpoint authentication requirement must be an explicit boolean")
    state = Path(state).absolute()
    if state.resolve(strict=True) != state or not state.is_dir():
        raise ValueError("ready task context state must be its exact existing canonical directory")
    prepared = preparation._load_prepared(state)
    if not preparation._is_multitask(prepared):
        raise ValueError("ready task context requires the explicit reviewed multi-task profile")
    repository = Path(prepared["repository"])
    if type(artifact) is not str:
        raise ValueError("ready task receipt requires an exact runtime artifact and trusted digest")
    relative = Path(artifact)
    path = repository / relative
    if (type(artifact) is not str or relative.is_absolute() or relative.as_posix() != artifact
            or ".." in relative.parts or not path.is_relative_to(repository / ".runtime")
            or path.resolve(strict=True) != path or path.is_symlink()
            or type(expected_sha256) is not str or len(expected_sha256) != 64
            or any(char not in "0123456789abcdef" for char in expected_sha256)):
        raise ValueError("ready task receipt requires an exact runtime artifact and trusted digest")
    raw = _read_regular(path, MAX_RESULT_BYTES)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("ready task receipt bytes or file identity differ")
    result = json.loads(raw, object_pairs_hook=retrieval_unique)
    fields = {
        "schema", "repository", "state", "task_cids", "contexts", "context_bundle", "manifest_cid",
        "admission_cid", "requirement_contract_cid", "native_binding", "source_hashes", "retrieval_reuse",
        "ir_selection_mode", "ir_catalog_path", "ir_selections", "ir_metadata_nominations",
        "checkpoint_bytes_authenticated", "decoder_runtime_admitted", "new_embedding_calls",
        "model_loading_calls", "training_steps", "provider_calls", "query_semantic_alignment_verified",
        "proof_authority", "execution_authority", "completion_authority", "canonical_task_mutated",
    }
    authenticated = type(result) is dict and result.get("schema") == CHECKPOINT_SCHEMA
    if authenticated:
        fields.add("ir_checkpoint_observations")
    if require_checkpoint_authentication and not authenticated:
        raise ValueError("ready task receipt requires authenticated original checkpoint observations")
    if (type(result) is not dict or set(result) != fields or result["schema"] not in (SCHEMA, CHECKPOINT_SCHEMA)
            or _wire(result) != raw or result["repository"] != str(repository) or result["state"] != str(state)
            or result["checkpoint_bytes_authenticated"] is not authenticated
            or any(result[name] is not False for name in (
                "decoder_runtime_admitted", "query_semantic_alignment_verified",
                "proof_authority", "execution_authority", "completion_authority", "canonical_task_mutated"))
            or any(type(result[name]) is not int or result[name] != 0 for name in (
                "new_embedding_calls", "model_loading_calls", "training_steps", "provider_calls"))):
        raise ValueError("ready task receipt is not the bounded canonical nomination")
    selected = result["task_cids"]
    if (type(selected) is not list or not 1 <= len(selected) <= MAX_TASKS
            or any(type(cid) is not str or not cid for cid in selected)
            or selected != sorted(set(selected)) or type(result["contexts"]) is not list
            or any(type(item) is not dict for item in result["contexts"])
            or [item.get("task_cid") for item in result["contexts"]] != selected):
        raise ValueError("ready task receipt selection differs from exact unique task contexts")
    if (type(result["context_bundle"]) is not dict
            or set(result["context_bundle"]) != {"artifact", "sha256"}
            or any(type(value) is not str for value in result["context_bundle"].values())):
        raise ValueError("ready task receipt bundle reference differs from the closed nomination")
    admission, admission_identity = _admission(state)
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    profile = local._verify_multitask_profile_sources(verified["manifest"])
    if (admission["manifest"] != prepared["manifest"] or profile is None
            or local.content_identity(admission["manifest"]) != result["manifest_cid"]
            or local.content_identity(admission) != result["admission_cid"]
            or profile["intent_requirement_contract_cid"] != result["requirement_contract_cid"]
            or result["source_hashes"] != {name: row["sha256"] for name, row in verified["manifest"]["sources"].items()}):
        raise ValueError("ready task receipt source or admission identities are stale")
    _source_inputs(repository, prepared)
    database = state / "intent.duckdb"
    database_identity = _database_identity(database)
    if (type(result["native_binding"]) is not dict
            or result["native_binding"].get("database") != database_identity):
        raise ValueError("ready task receipt differs from the exact existing file-backed owner")
    ir_path, ir_requests = result["ir_catalog_path"], result["ir_selections"]
    if authenticated and (ir_path is None or ir_requests is None):
        raise ValueError("authenticated checkpoint receipt requires exact per-task IR nominations")
    if (ir_path is None) != (ir_requests is None):
        raise ValueError("ready task receipt IR metadata nomination is incomplete")
    if ir_path is not None and type(ir_path) is not str:
        raise ValueError("ready task receipt IR catalog path differs from the exact nomination")
    if ir_requests is not None and (type(ir_requests) is not dict or set(ir_requests) != set(selected)):
        raise ValueError("ready task receipt IR metadata task identities differ")
    catalog = Path(ir_path) if ir_path is not None else None
    with IntentRepository(database, install_schema=False) as intent:
        native = _native_observation(intent, admission, verified, selected)
        if result["native_binding"] != {"database": database_identity, **native}:
            raise ValueError("ready task receipt native task revision, dependency or world binding is stale")
        ir_current = _resolve_ir(catalog, ir_requests, selected)
        if (ir_current != result["ir_metadata_nominations"]
                or result["ir_selection_mode"] != ("authenticated_checkpoint_nomination" if authenticated else
                    "metadata_nomination" if ir_current is not None else "structural_source_context")):
            raise ValueError("ready task receipt IR catalog generation or exact selection changed")
        if authenticated and _wire(_authenticate_ir(catalog, ir_requests, selected, ir_current)) != _wire(result["ir_checkpoint_observations"]):
            raise ValueError("ready task receipt original checkpoint bytes or file identities changed")
        _verify_plain_bundle(repository, result["context_bundle"], selected)
        for context in result["contexts"]:
            task = next(row for row in native["tasks"] if row["task_cid"] == context["task_cid"])
            if (context.get("schema") != "supervisor-task-context-preparation@1"
                    or context.get("task_id") != task["task_id"] or context.get("task_revision") != task["revision"]
                    or context.get("event_watermark") != native["event_watermark"]
                    or context.get("task_title") != next(row.objective for row in verified["graph"].tasks
                                                         if row.task_cid == context["task_cid"])):
                raise ValueError("ready task context differs from the exact native task identity")
            _verify_context(repository, intent, context, prepared, profile)
            selection = load_task_context_selection(repository=repository,
                artifact=result["context_bundle"]["artifact"], expected_sha256=result["context_bundle"]["sha256"],
                task_cid=context["task_cid"], task_id=context["task_id"])
            if selection["metadata"] != {name.lower(): value for name, value in context["metadata"].items()}:
                raise ValueError("ready task context bundle differs from the sealed task nomination")
            if _retrieval_binding(repository, context) != result["retrieval_reuse"]:
                raise ValueError("ready task receipt differs from the exact reused retrieval configuration")
        if authenticated and _wire(_authenticate_ir(catalog, ir_requests, selected, ir_current)) != _wire(result["ir_checkpoint_observations"]):
            raise ValueError("ready task checkpoint owners changed during cold reopening")
        if (_native_observation(intent, admission, verified, selected) != native
                or preparation._load_prepared(state) != prepared
                or _admission(state)[1] != admission_identity
                or _resolve_ir(catalog, ir_requests, selected) != ir_current
                or _database_identity(database) != result["native_binding"]["database"]):
            raise ValueError("ready task context owners changed during cold reopening")
        if authenticated:
            _checkpoint_file_fence(result["ir_checkpoint_observations"])
    return result


def retrieval_unique(pairs):
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate retained admission key")
        result[name] = value
    return result
