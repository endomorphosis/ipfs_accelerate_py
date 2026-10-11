"""Consume shared learned NumPy advice from captured, declared source bytes.

The shared datasets owner controls embeddings, isolated inference and its
immutable operation. This consumer binds that observation to planner/worker
evidence and native world/metadata lakes. Source alignment is not a proof.
"""
import json
import math
from pathlib import Path
import stat
import time

from ..proof.formal_verification_contracts import content_identity
from ..semantic_state.program_world_database import ProgramWorldDatabase
from .supervisor_meta_index import SupervisorMetaIndex
from .security_autoencoder_advisor import _bytes, _sha, _read, _write, _hydrate_lakes
from . import source384_repository_context as source_owner
from . import numerical_selection_config as config_owner

SCHEMA = "supervisor-learned-numerical-selection@1"
PROFILE = config_owner.PROFILE
OPERATION = "learned_numerical_selection_observation"
MAX_INFERENCE_BYTES = 32 * 1024**2
AUTHORITY = dict(proof_authority=False, formalization_authority=False, planning_authority=False,
    dispatch_authority=False, execution_authority=False, completion_authority=False,
    source_semantics_verified=False, whole_program_semantics_verified=False,
    source_obligations_discharged=0, training_steps=0, provider_calls=0, download_calls=0)
_sources = source_owner._sources


def _producers():
    from ipfs_datasets_py.logic.software_contracts import codebase_numerical_selection_384 as native
    return dict(consumer=_sha(_read(Path(__file__).resolve())), config=_sha(_read(Path(config_owner.__file__).resolve())),
        source_capture=source_owner._pins(), native=native.pins())


def _files(output):
    # Mutable source/model DBs are checked by their native logical owners.
    # Bind retained CAS, receipts and projected lake bytes independently.
    result, total = {}, 0
    for path in sorted(output.rglob("*")):
        if path.name in {"receipt.json", "source.duckdb", "models.duckdb"} and path.parent == output:
            continue
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        raw = _read(path, MAX_INFERENCE_BYTES)
        total += len(raw)
        if total > 256 * 1024**2 or len(result) >= 2048:
            raise ValueError("numerical selection artifact inventory exceeds bound")
        result[path.relative_to(output).as_posix()] = dict(sha256=_sha(raw), bytes=len(raw))
    return result


def _paths(paths, hashes, inventory):
    if paths is None:
        paths = sorted(name for name in hashes if name.endswith(".py"))
    if (type(paths) not in (list, tuple) or not 0 < len(paths) <= 256
            or any(type(p) is not str or p not in hashes for p in paths)
            or len(set(paths)) != len(paths)):
        raise ValueError("explicit bounded declared numerical program population required")
    captured = {row["path"] for row in inventory if row["disposition"] == "captured"}
    if not set(paths) <= captured:
        raise ValueError("selected numerical program input was not captured")
    return sorted(paths)


def _summary(inference, *, checkpoint_file_sha256, inference_sha256):
    report = inference["report"]
    output = report["output"]
    rows = [row for row in output["rows"] if row["candidate"] is not None]
    samples = []
    for row in sorted(rows, key=lambda r: (r["status"] != "source_aligned_fragment", r["path"], r["unit_id"])):
        candidate = row["candidate"]
        refs = candidate["matched_projection_refs"]
        refs = ({key: refs[key] for key in ("numerical_report_sha256", "link_report_sha256",
            "pending_obligation_ids", "family_artifact_ids")} if refs is not None else None)
        item = dict(path=row["path"], unit_id=row["unit_id"], source_sha256=row["source_sha256"],
            status=row["status"], candidate_ir_json=_bytes(candidate["prediction"]["candidate_ir"]).decode(),
            source_binding_json=_bytes(candidate["source_binding"]).decode(),
            matched_projection_refs_json=_bytes(refs).decode(),
            projection_artifact_bodies_in_retained_inference=True, candidate_replaced=False)
        if len(samples) < 2 and len(_bytes(item)) <= 3000:
            samples.append(item)
    result = dict(schema="supervisor-learned-numerical-selection-summary@1", profile=PROFILE,
        origin="learned_checkpoint_prediction_with_independent_source_alignment",
        checkpoint_file_sha256=checkpoint_file_sha256,
        checkpoint_canonical_sha256=report["key"]["checkpoint_canonical_sha256"],
        head_sha256=report["key"]["head_sha256"], projection_sha256=report["key"]["projection_sha256"],
        inference_sha256=inference_sha256, coverage_json=_bytes(report["coverage"]).decode(),
        candidates=samples, total_candidates=len(rows), omitted_candidates=len(rows)-len(samples),
        neural_inference_replayed=False, model_loads=output["model_loads"],
        interpretation="Experimental unary selection CALL advice; imports and source spans retained; pending obligations are not proofs.",
        **AUTHORITY)
    while len(_bytes(result)) > 8192 and result["candidates"]:
        result["candidates"].pop()
        result["omitted_candidates"] += 1
    if len(_bytes(result)) > 8192:
        raise ValueError("learned numerical summary exceeds bound")
    return result


def _catalogs(output, record, world_cid):
    return [("vector", str(output / "inference.json"), "sha256:"+record["inference_sha256"]),
        ("knowledge_graph", str(output / "planning-summary.json"), "sha256:"+record["summary_sha256"]),
        ("world_model", str(output / "world.duckdb"), world_cid)]


def _hydrate(output, record):
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    found = world.records_for_decision(task_id=record["task_id"], operation=OPERATION)
    if found["n"] != 1 or found["records"][0]["payload"] != record:
        raise ValueError("learned numerical world observation differs")
    cid = found["records"][0]["record_cid"]
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    linked = meta.compose_for_subject(subject_kind="tree_id", subject_ref=record["source_tree_id"])
    expected = set(_catalogs(output, record, cid))
    if linked["n"] != len(expected) or {(r["catalog_kind"],r["locator_ref"],r["record_ref"]) for r in linked["linked"]} != expected:
        raise ValueError("learned numerical metadata differs")
    return dict(world_record_cid=cid, catalog_count=len(expected), **_hydrate_lakes(output, record, cid))


def _record(root, hashes, selected, producers, config_sha256, inference_sha256, summary):
    tree = content_identity(dict(schema=SCHEMA, sources=hashes, selected_paths=selected))
    return dict(task_id="numerical-selection:"+tree, board="learned-numerical-selection", operation=OPERATION,
        repository=str(root), source_tree_id=tree, source_hashes=hashes, selected_paths=selected,
        producers=producers, config_sha256=config_sha256, inference_sha256=inference_sha256,
        summary_sha256=_sha(_bytes(summary)), proposal_only=True, cas_completed=False, admitted=False, **AUTHORITY)


def _deadline(timeout, budget_profile=None):
    from .benchmark_budget_policy import source384_timeout_limit
    maximum = source384_timeout_limit(budget_profile)
    if type(timeout) not in (int,float) or not math.isfinite(timeout) or not 0 < timeout <= maximum:
        raise ValueError("bounded numerical selection deadline required")
    end = time.monotonic()+timeout
    def remaining():
        left = end-time.monotonic()
        if left <= 0:
            raise TimeoutError("numerical selection deadline expired")
        return left
    return remaining


def prepare_numerical_selection_context(*, repository, source_hashes, output, config_path,
        expected_config_sha256, program_paths=None, timeout_seconds=180., memory_mb=4096,
        scheduler=None, parent_lease=None, cancel_event=None, budget_profile=None):
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources, codebase_admission_timeout
    from ipfs_datasets_py.logic.software_contracts.codebase_numerical_selection_384 import infer_numerical_selection
    remaining = _deadline(timeout_seconds, budget_profile)
    root = Path(repository).resolve(strict=True)
    output = source_owner._output(root, output, fresh=True)
    config_path = Path(config_path)
    config = config_owner.load_numerical_selection_config(config_path, expected_config_sha256)
    hashes = _sources(root, source_hashes)
    producers = _producers()
    output.mkdir(parents=True, mode=0o700)
    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease, cancel_event=cancel_event,
            timeout_seconds=codebase_admission_timeout(remaining_seconds=remaining()), memory_mb=memory_mb+2048,
            cpu_slots=3, child_process_slots=3) as lease, source_owner._owners(output) as (index, registry):
        head = index.prepare_current(root, repository_id="numerical-selection:"+content_identity(dict(repository=str(root), sources=hashes)),
            operation_id="initial", expected_head=None,
            limits=CodebaseScanLimits(max_entries=len(hashes), max_file_bytes=1024*1024), exclusions=[".runtime"],
            parent_lease=lease, cancel_event=cancel_event, timeout_seconds=remaining(), memory_mb=memory_mb).head
        inventory = source_owner._inventory(index, head, hashes)
        selected = _paths(program_paths, hashes, inventory)
        inference = infer_numerical_selection(index, root, expected_head=head, registry=registry,
            checkpoint_path=config["checkpoint_path"], checkpoint_sha256=config["checkpoint_sha256"],
            paths=selected, embedding_snapshot=config["embedding_snapshot"], parent_lease=lease,
            cancel_event=cancel_event, timeout_seconds=remaining(), memory_mb=memory_mb,
            **({"budget_profile": budget_profile} if budget_profile is not None else {}))
    if len(_bytes(inference)) > MAX_INFERENCE_BYTES:
        raise ValueError("learned numerical inference exceeds consumer bound")
    digest = _write(output / "inference.json", inference)
    summary = _summary(inference, checkpoint_file_sha256=config["checkpoint_sha256"], inference_sha256=digest)
    _write(output / "planning-summary.json", summary)
    record = _record(root, hashes, selected, producers, expected_config_sha256, digest, summary)
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    persisted = world.persist(record)
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    for kind, locator, ref in _catalogs(output, record, persisted["record_cid"]):
        catalog = meta.register_catalog(kind=kind, locator_ref=locator, repository_id=content_identity(dict(repository=str(root))),
            tree_id=record["source_tree_id"], project=False)
        meta.link_identity(subject_kind="tree_id", subject_ref=record["source_tree_id"], catalog_id=catalog["catalog_id"],
            record_kind="learned_numerical_"+kind, record_ref=ref, project=False)
    projected = meta.project_ducklake()
    if persisted["ducklake"]["status"] != "projected" or projected["status"] != "projected":
        raise ValueError("learned numerical DuckLake projection unavailable")
    receipt = dict(schema=SCHEMA, profile=PROFILE, repository=str(root), output=str(output), source_hashes=hashes,
        selected_paths=selected, source_head=head.to_dict(), source_inventory=inventory, producers=producers,
        config_path=str(config_path), config_sha256=expected_config_sha256,
        checkpoint_file_sha256=config["checkpoint_sha256"], inference_sha256=digest,
        summary=summary, record=record, hydration=_hydrate(output, record),
        inference_execution=dict(native_worker_executed=inference["native_worker_executed"],
            inference_executed=inference["inference_executed"], model_loads=inference["report"]["output"]["model_loads"]),
        neural_inference_replayed=False, files=_files(output), **AUTHORITY)
    _sources(root, hashes)
    if producers != _producers() or config != config_owner.load_numerical_selection_config(config_path, expected_config_sha256):
        raise ValueError("numerical selection source or assets changed during preparation")
    remaining()
    if len(_bytes(receipt)) > 131072:
        raise ValueError("learned numerical receipt exceeds consumer bound")
    return dict(receipt, receipt_sha256=_write(output / "receipt.json", receipt))


def validate_historical_numerical_selection_context(*, repository, expected_receipt):
    root = Path(repository).resolve(strict=True)
    receipt = expected_receipt
    fields = {"schema","profile","repository","output","source_hashes","selected_paths","source_head","source_inventory",
        "producers","config_path","config_sha256","checkpoint_file_sha256","inference_sha256","summary","record",
        "hydration","inference_execution","neural_inference_replayed","files","receipt_sha256",*AUTHORITY}
    if (type(receipt) is not dict or set(receipt) != fields or receipt["schema"] != SCHEMA
            or receipt["profile"] != PROFILE or receipt["repository"] != str(root)
            or receipt["neural_inference_replayed"] is not False
            or any(type(receipt[k]) is not type(v) or receipt[k] != v for k,v in AUTHORITY.items())):
        raise ValueError("closed learned numerical receipt required")
    output = source_owner._output(root, receipt["output"])
    raw = _read(output / "receipt.json", 131072)
    if raw != _bytes({k:v for k,v in receipt.items() if k != "receipt_sha256"}) or _sha(raw) != receipt["receipt_sha256"]:
        raise ValueError("learned numerical receipt bytes differ")
    config = config_owner.load_numerical_selection_config(receipt["config_path"], receipt["config_sha256"])
    if (config["checkpoint_sha256"] != receipt["checkpoint_file_sha256"] or _producers() != receipt["producers"]
            or _files(output) != receipt["files"] or _sha(_read(output / "inference.json", MAX_INFERENCE_BYTES)) != receipt["inference_sha256"]):
        raise ValueError("learned numerical historical assets or artifacts differ")
    return receipt


def validate_numerical_selection_context(*, repository, expected_receipt, timeout_seconds=180.,
        memory_mb=4096, scheduler=None, parent_lease=None, cancel_event=None, budget_profile=None):
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_numerical_selection_384 import validate_numerical_selection
    remaining = _deadline(timeout_seconds, budget_profile)
    receipt = validate_historical_numerical_selection_context(repository=repository, expected_receipt=expected_receipt)
    root = Path(repository).resolve(strict=True)
    hashes = _sources(root, receipt["source_hashes"])
    output = source_owner._output(root, receipt["output"])
    config = config_owner.load_numerical_selection_config(receipt["config_path"], receipt["config_sha256"])
    inference = json.loads(_read(output / "inference.json", MAX_INFERENCE_BYTES))
    key = inference["report"]["key"]
    if key["source_head"] != receipt["source_head"] or key["checkpoint_file_sha256"] != receipt["checkpoint_file_sha256"]:
        raise ValueError("learned numerical inference source/model selection differs")
    with source_owner._owners(output) as (index, registry):
        inventory = source_owner._inventory(index, CodebaseHead.from_dict(receipt["source_head"]), hashes)
        if (inventory != receipt["source_inventory"] or _paths(receipt["selected_paths"], hashes, inventory) != receipt["selected_paths"]
                or inference["report"]["preparation"]["paths"] != receipt["selected_paths"]):
            raise ValueError("learned numerical selected source population differs")
        validate_numerical_selection(index, root, inference, registry=registry,
            checkpoint_path=config["checkpoint_path"], checkpoint_sha256=config["checkpoint_sha256"],
            embedding_snapshot=config["embedding_snapshot"], scheduler=scheduler, parent_lease=parent_lease,
            cancel_event=cancel_event, timeout_seconds=remaining(), memory_mb=memory_mb,
            **({"budget_profile": budget_profile} if budget_profile is not None else {}))
    summary = _summary(inference, checkpoint_file_sha256=receipt["checkpoint_file_sha256"], inference_sha256=receipt["inference_sha256"])
    record = _record(root, hashes, receipt["selected_paths"], receipt["producers"], receipt["config_sha256"], receipt["inference_sha256"], summary)
    execution = dict(native_worker_executed=inference["native_worker_executed"],
        inference_executed=inference["inference_executed"], model_loads=inference["report"]["output"]["model_loads"])
    if (_bytes(summary) != _bytes(receipt["summary"]) or _bytes(record) != _bytes(receipt["record"])
            or _read(output / "planning-summary.json") != _bytes(summary)
            or _bytes(execution) != _bytes(receipt["inference_execution"]) or _hydrate(output, record) != receipt["hydration"]):
        raise ValueError("learned numerical summary, execution or lake observation differs")
    _sources(root, hashes)
    validate_historical_numerical_selection_context(repository=root, expected_receipt=receipt)
    remaining()
    return receipt
