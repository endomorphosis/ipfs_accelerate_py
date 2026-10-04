"""Pinned Source384 advice from the signed repository population.

The datasets owners retain the structural index, exact function maps, model
version and numerical receipts. This consumer never trains or grants proof
authority. Reloading observes current source and assets without neural replay.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

from .security_autoencoder_advisor import _read
from .source384_config import load_source384_config
from . import source384_program_scope as scope_owner

SCHEMA = "terminal-source384-repository-context@1"
HEADER_SCHEMA = "terminal-source384-repository-context@2"
SUCCESSOR_SCHEMA = "terminal-source384-repository-context@3"
MAX_RECEIPT_BYTES = 131072
MAX_INFERENCE_BYTES = 32 * 1024 * 1024
AUTHORITY = dict(proof_authority=False, execution_authority=False,
                 completion_authority=False, formalization_authority=False)
SUCCESSOR_RECEIPT_FIELDS = frozenset({"schema", "output", "repository", "source_hashes", "source_head",
    "source_inventory", "config_path", "config_sha256", "checkpoint_sha256", "version_id",
    "inference_sha256", "producer", "summary", "seconds", "training_steps", "neural_inference_replayed",
    "resource_profile", "successor_lineage", "requires_independent_manifest", "planning_authority",
    "dispatch_authority", "inference_execution", *AUTHORITY})


def _validate_successor_envelope(receipt):
    """Close the advisory envelope independently of any source-current replay."""
    _require(type(receipt) is dict and set(receipt) == SUCCESSOR_RECEIPT_FIELDS | ({"program_scope"} if "program_scope" in receipt else set())
             and receipt["schema"] == SUCCESSOR_SCHEMA
             and receipt["requires_independent_manifest"] is True
             and receipt["planning_authority"] is False and receipt["dispatch_authority"] is False
             and receipt["neural_inference_replayed"] is False
             and type(receipt["training_steps"]) is int and receipt["training_steps"] == 0
             and all(receipt[key] is False for key in AUTHORITY),
             "closed advisory Source384 successor envelope required")
    execution = receipt["inference_execution"]
    _require(type(execution) is dict and set(execution) ==
             {"native_worker_executed", "inference_executed", "model_loads"}
             and execution["native_worker_executed"] is True and execution["inference_executed"] is True
             and type(execution["model_loads"]) is int and 0 <= execution["model_loads"] <= 1024,
             "bounded exact Source384 inference observations required")
    lineage = receipt["successor_lineage"]
    _require(type(lineage) is dict and set(lineage) == {"schema", "predecessor_output",
             "predecessor_receipt_sha256", "predecessor_source_head", "predecessor_inference_sha256",
             "source_paths", "excluded_new_output_paths", "requires_independent_manifest",
             "historical_currentness_verified", "header_nomination_inherited"}
             and lineage["schema"] == "source384-advisory-successor-lineage@1"
             and lineage["requires_independent_manifest"] is True
             and lineage["historical_currentness_verified"] is False
             and lineage["header_nomination_inherited"] is False
             and lineage["source_paths"] == sorted(receipt["source_hashes"])
             and lineage["predecessor_output"] != receipt["output"],
             "closed bounded Source384 predecessor lineage required")
    for name in ("predecessor_receipt_sha256", "predecessor_inference_sha256"):
        value = lineage[name]
        _require(type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
                 "exact Source384 historical digest required")
    _output(Path(receipt["repository"]), lineage["predecessor_output"])
    _successor_exclusions(lineage["excluded_new_output_paths"], receipt["source_hashes"])


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _remaining(deadline):
    left = deadline - time.monotonic()
    _require(left > 0, "Source384 operation deadline expired")
    return left


def _pins():
    from . import source384_config, security_autoencoder_advisor, terminal_source_partition, terminal_task_profile
    from ipfs_datasets_py.duckdb_control import codebase_catalog
    from ipfs_datasets_py.logic.software_contracts import (
        codebase_ir, duckdb_ast_store, duckdb_ingest, content, python_frontend,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index import scanner, snapshot, python_analysis
    return {"consumer": _sha(Path(__file__).read_bytes()),
            "config": _sha(Path(source384_config.__file__).read_bytes()),
            "byte_reader": _sha(Path(security_autoencoder_advisor.__file__).read_bytes()),
            "source_owners": {module.__name__: _sha(Path(module.__file__).read_bytes())
                for module in (codebase_catalog, codebase_ir, duckdb_ast_store, duckdb_ingest,
                               scanner, snapshot, python_analysis, content, python_frontend,
                               scope_owner, terminal_source_partition, terminal_task_profile)}}


def _write(path, value, maximum):
    raw = _raw(value)
    _require(len(raw) <= maximum, "Source384 artifact exceeds its byte bound")
    with path.open("xb") as stream:
        stream.write(raw)
    return _sha(raw)


def _sources(repository, source_hashes):
    # Match the signed create-output manifest's fixed256-input bound. The
    # original public checkout has218 files before supervisor inputs.
    _require(type(source_hashes) is dict and 1 <= len(source_hashes) <= 256,
             "complete bounded signed Source384 source population required")
    result = {}
    for name, digest in sorted(source_hashes.items()):
        path = Path(name)
        _require(type(name) is str and path.as_posix() == name and not path.is_absolute()
                 and name not in {".", ".."} and ".." not in path.parts
                 and not any(part in {".git", ".runtime"} for part in path.parts),
                 "canonical authorized Source384 source path required")
        _require(type(digest) is str and len(digest) == 64,
                 "exact signed source digest required")
        raw = _read(repository / name, 1024 * 1024)
        _require(_sha(raw) == digest, "Source384 source differs from signed population")
        result[name] = digest
    return result


def _output(repository, output, *, fresh=False):
    output = Path(output)
    _require(output.is_absolute() and output.resolve() == output
             and not output.is_relative_to(repository) and not repository.is_relative_to(output)
             and (not fresh or not output.exists()),
             "canonical private external Source384 state required")
    return output


@contextmanager
def _owners(output):
    import duckdb
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseCatalog
    from ipfs_datasets_py.logic.software_contracts.cache import ImmutableCAS
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import RepositoryCodebaseIndex
    from ipfs_datasets_py.logic.software_contracts.duckdb_ast_store import DuckDBASTStore
    from ipfs_datasets_py.logic.software_contracts.duckdb_ingest import DuckDBASTIngestor
    with duckdb.connect(str(output / "source.duckdb"), config={"threads": 1, "memory_limit": "512MB"}) as connection:
        store = DuckDBASTStore(connection=connection)
        artifacts = ImmutableCAS(output / "source-artifacts")
        index = RepositoryCodebaseIndex(ingestor=DuckDBASTIngestor(store=store),
            artifacts=artifacts, catalog=CodebaseCatalog(store, artifacts))
        with AutoencoderRegistry(output / "models.duckdb", output / "model-artifacts",
                                 max_artifact_bytes=MAX_INFERENCE_BYTES) as registry:
            yield index, registry


def _inventory(index, head, source_hashes):
    """Account for scanner exclusions; never admit undeclared captured files."""
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import _ignored_raw
    manifest = index.load(head.manifest_cid)
    captured = {entry.path: entry for entry in manifest.snapshot.entries}
    _require(set(captured) <= set(source_hashes), "Source384 index contains undeclared source")
    exclusions = tuple(value.encode() for value in manifest.snapshot.exclusions)
    rows = []
    for name, digest in sorted(source_hashes.items()):
        entry = captured.get(name)
        if entry is None:
            _require(_ignored_raw(name.encode(), exclusions), "signed Source384 source is missing from index")
            rows.append(dict(path=name, sha256=digest, disposition="excluded_by_native_scan_policy"))
        else:
            _require(entry.source_cid is not None and not entry.is_opaque,
                     "signed Source384 source could not be captured exactly")
            _require(_sha(index.artifacts.get_bytes(entry.source_cid)) == digest,
                     "Source384 captured bytes differ from signed source")
            rows.append(dict(path=name, sha256=digest, source_cid=entry.source_cid,
                             disposition="captured"))
    return rows


def _summary(inference, *, inventory, checkpoint_sha256, inference_sha256, advisory_successor=False,
             program_scope=None):
    # The full native receipt is separately retained. Coverage is never inferred
    # from a truncated sample, nor interpreted as a checked property.
    report = inference["report"]
    units = {unit["unit_id"]: (file["path"], unit) for file in report["preparation"]["files"]
             if file.get("extraction") for unit in file["extraction"]["units"]}
    candidates = []
    total_candidates = sum(row["candidate"] is not None for row in report["output"]["rows"])
    for row in report["output"]["rows"]:
        if row["candidate"] is None or len(candidates) >= 2:
            continue
        path, unit = units[row["id"]]
        candidate = row["candidate"]
        validation = candidate.get("source_contract") or {}
        # Numerical vectors and full line maps belong in the retained native
        # artifact. Including them here would omit every candidate by size.
        item = dict(unit_id=row["id"], path=path, qualified_name=unit["qualified_name"],
            source_sha256=unit["source_sha256"],
            start_byte=unit["start_byte"], end_byte=unit["end_byte"],
            normalized_body_sha256=unit["normalized_body_sha256"],
            source_map_sha256=_sha(_raw(unit["source_binding"])),
            # Canonical JSON keeps the complete typed IR within the planner's
            # existing nesting bound; the normal text and byte checks still
            # apply. The enclosing native candidate remains separately bound.
            candidate_sha256=_sha(_raw(candidate)),
            candidate_ir_json=_raw(candidate.get("candidate_ir")).decode("utf-8"),
            status=candidate["status"], source_validation_status=validation.get("status", "unavailable"),
            source_validation_reason=validation.get("reason"),
            source_semantics_verified=False, **AUTHORITY)
        if len(_raw(item)) <= 2048:
            candidates.append(item)
    result = dict(schema="terminal-source384-planning-summary@1", mode="pinned_parent",
        checkpoint_sha256=checkpoint_sha256, inference_sha256=inference_sha256,
        source_head=report["key"]["source_head"], version_id=report["key"]["version_id"],
        source_files=len(inventory), excluded_source_files=sum(
            row["disposition"] != "captured" for row in inventory),
        coverage=report["coverage"], repository_retention="not_evaluated",
        candidate_samples=candidates, omitted_candidates=total_candidates-len(candidates),
        training_steps=0, provider_calls=0, model_promotion_performed=False,
        nomination_only=True, **AUTHORITY)
    if program_scope is not None:
        result.update(program_scope_schema=program_scope["schema"],
            program_scope_sha256=_sha(_raw(program_scope)),
            program_source_files=len(program_scope["program_paths"]),
            harness_support_files=len(program_scope["harness_support"]),
            inference_python_files=len(report["preparation"]["paths"]))
    if advisory_successor:
        result.update(advice_only=True, requires_independent_manifest=True,
                      planning_authority=False, dispatch_authority=False)
    _require(len(_raw(result)) <= 8192, "Source384 planning summary exceeds fixed bound")
    return result


def _validate_population(inference, inventory, program_scope=None):
    preparation = inference["report"]["preparation"]
    paths = _program_python_paths(inventory, program_scope)
    _require(preparation["paths"] == paths and preparation["max_functions"] == 1024
             and preparation["max_selected_units"] == 128,
             "Source384 inference selection differs from the complete declared Python population")


def _program_python_paths(inventory, program_scope=None):
    program = None if program_scope is None else set(program_scope["program_paths"])
    return sorted(row["path"] for row in inventory if row["disposition"] == "captured"
                  and row["path"].endswith(".py") and (program is None or row["path"] in program))


def _scope_selection(root, output, source_hashes, expected_scope):
    scope_owner.require_selected_scope(source_hashes, expected_scope)
    if expected_scope is None:
        return None
    raw = _read(output / scope_owner.ARTIFACT, scope_owner.MAX_BYTES)
    _require(_sha(raw) == expected_scope["selection_sha256"], "Source384 selection manifest bytes changed")
    envelope = json.loads(raw)
    _require(raw == _raw(envelope), "canonical Source384 selection artifact required")
    scope = scope_owner.program_scope(repository=root, source_hashes=source_hashes, envelope=envelope)
    _require(scope == expected_scope, "Source384 program scope changed")
    return envelope


def prepare_source384_context(*, repository, source_hashes, output, config_path,
                              timeout_seconds=90., scheduler=None, parent_lease=None, intent_binding=None,
                              manifest_envelope=None):
    return _prepare_source384_context(repository=repository, source_hashes=source_hashes,
        output=output, config_path=config_path, timeout_seconds=timeout_seconds,
        scheduler=scheduler, parent_lease=parent_lease, intent_binding=intent_binding,
        manifest_envelope=manifest_envelope)


def validate_historical_source384_selection(*, repository, expected_receipt):
    """Verify selected historical bytes/assets, without claiming source currentness."""
    root = Path(repository).resolve(strict=True)
    _require(type(expected_receipt) is dict and expected_receipt.get("schema") in
             {SCHEMA, HEADER_SCHEMA, SUCCESSOR_SCHEMA}, "historical Source384 selection required")
    output = _output(root, expected_receipt["output"])
    receipt = json.loads(_read(output / "receipt.json", MAX_RECEIPT_BYTES))
    if receipt.get("schema") == SUCCESSOR_SCHEMA:
        _validate_successor_envelope(receipt)
    _require(receipt == expected_receipt and receipt["repository"] == str(root)
             and receipt["producer"] == _pins() and receipt["training_steps"] == 0
             and all(receipt.get(key) is False for key in AUTHORITY),
             "historical Source384 receipt or producer changed")
    config_path = Path(receipt["config_path"])
    config = load_source384_config(config_path)
    _require(_sha(_read(config_path, 131072)) == receipt["config_sha256"]
             and config["checkpoint_sha256"] == receipt["checkpoint_sha256"],
             "historical Source384 selected configuration changed")
    _require(_sha(_read(output / "inference.json", MAX_INFERENCE_BYTES)) == receipt["inference_sha256"],
             "historical Source384 inference bytes changed")
    _require(receipt["schema"] != HEADER_SCHEMA or receipt.get("header_consumer_sha256") == _header_pin(),
             "historical Source384 header producer changed")
    if "program_scope" in receipt:
        raw = _read(output / scope_owner.ARTIFACT, scope_owner.MAX_BYTES)
        scope = receipt["program_scope"]
        _require(_sha(raw) == scope["selection_sha256"], "historical Source384 selection bytes changed")
        envelope = json.loads(raw)
        scope_owner.verified_selection_manifest(envelope, repository=root)
        _require(raw == _raw(envelope) and scope == scope_owner.recorded_scope(
            source_hashes=receipt["source_hashes"], envelope=envelope),
            "historical Source384 selection identity changed")
    scope_owner.require_selected_scope(receipt["source_hashes"], receipt.get("program_scope"))
    return receipt


def _successor_lineage(predecessor, excluded_paths):
    return dict(schema="source384-advisory-successor-lineage@1",
        predecessor_output=predecessor["output"],
        predecessor_receipt_sha256=_sha(_raw(predecessor)),
        predecessor_source_head=predecessor["source_head"],
        predecessor_inference_sha256=predecessor["inference_sha256"],
        source_paths=sorted(predecessor["source_hashes"]),
        excluded_new_output_paths=list(excluded_paths),
        requires_independent_manifest=True, historical_currentness_verified=False,
        header_nomination_inherited=False)


def _successor_exclusions(paths, source_hashes):
    from ipfs_datasets_py.logic.software_contracts.semantic_index.snapshot import _ignored_raw
    _require(type(paths) in (list, tuple) and len(paths) <= 256
             and all(type(name) is str for name in paths)
             and list(paths) == sorted(set(paths)), "bounded exact successor exclusions required")
    for name in paths:
        path = Path(name)
        _require(path.as_posix() == name and not path.is_absolute() and name not in {".", ".."}
                 and ".." not in path.parts and name not in source_hashes
                 and not any(part in {".git", ".runtime"} for part in path.parts),
                 "successor exclusion must be outside unchanged source population")
    _require(not any(_ignored_raw(name.encode(), tuple(path.encode() for path in paths))
                     for name in source_hashes), "successor exclusions hide predecessor source population")
    return tuple(paths)


def prepare_source384_successor_context(*, repository, source_hashes, output, predecessor_receipt,
                                       timeout_seconds=90., scheduler=None, parent_lease=None,
                                       excluded_new_output_paths=()):
    """Infer current advice with exact predecessor assets; never inherit its proof nomination."""
    started = time.monotonic()
    _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 180,
             "bounded Source384 successor deadline required")
    predecessor = validate_historical_source384_selection(repository=repository,
        expected_receipt=predecessor_receipt)
    _require(set(source_hashes) == set(predecessor["source_hashes"]),
             "Source384 successor source population changed")
    exclusions = _successor_exclusions(excluded_new_output_paths, source_hashes)
    _require(str(output) != predecessor["output"], "fresh Source384 successor store required")
    return _prepare_source384_context(repository=repository, source_hashes=source_hashes,
        output=output, config_path=predecessor["config_path"],
        timeout_seconds=_remaining(started + timeout_seconds), scheduler=scheduler,
        parent_lease=parent_lease, predecessor=predecessor, excluded_new_output_paths=exclusions)


def _prepare_source384_context(*, repository, source_hashes, output, config_path,
                              timeout_seconds=90., scheduler=None, parent_lease=None, intent_binding=None,
                              predecessor=None, excluded_new_output_paths=(), manifest_envelope=None):
    started = time.monotonic()
    _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 180,
             "bounded Source384 preparation deadline required")
    def remaining():
        left = timeout_seconds - (time.monotonic() - started)
        _require(left > 0, "Source384 preparation deadline expired")
        return left
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources, codebase_admission_timeout
    from ipfs_datasets_py.logic.software_contracts.codebase_source_384 import register_shared_parent
    from ipfs_datasets_py.logic.software_contracts.codebase_source_units_384 import infer_shared_parent_units
    root = Path(repository).resolve(strict=True)
    output = _output(root, output, fresh=True)
    config_path = Path(config_path)
    source_hashes = _sources(root, source_hashes)
    producer = _pins()
    program_scope = None
    if predecessor is not None and "program_scope" in predecessor:
        previous_output = _output(root, predecessor["output"])
        # Original support stays immutable; current program hashes may differ.
        manifest_envelope = json.loads(_read(previous_output / scope_owner.ARTIFACT, scope_owner.MAX_BYTES))
        _require(_sha(_raw(manifest_envelope)) == predecessor["program_scope"]["selection_sha256"],
                 "Source384 predecessor selection changed")
        program_scope = scope_owner.program_scope(repository=root, source_hashes=source_hashes,
            envelope=manifest_envelope)
    elif manifest_envelope is not None and scope_owner.PROFILE in source_hashes:
        program_scope = scope_owner.program_scope(repository=root, source_hashes=source_hashes,
            envelope=manifest_envelope, current=True)
    scope_owner.require_selected_scope(source_hashes, program_scope)
    if program_scope is not None:
        _require(any(name.endswith(".py") for name in program_scope["program_paths"]),
                 "Source384 abstained: no declared Python program inputs; checkpoint not consumed")
    remaining()
    output.mkdir(mode=0o700)
    if program_scope is not None:
        _write(output / scope_owner.ARTIFACT, manifest_envelope, scope_owner.MAX_BYTES)
    # Bounds cover signed input only; the native scanner refuses larger trees.
    limits = CodebaseScanLimits(max_entries=len(source_hashes), max_file_bytes=1024 * 1024)
    repository_id = "terminal-source384:" + _sha(_raw(dict(repository=str(root), sources=source_hashes)))
    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease,
            timeout_seconds=codebase_admission_timeout(remaining_seconds=remaining()), memory_mb=6144, cpu_slots=3,
            child_process_slots=3) as lease, _owners(output) as (index, registry):
        config = load_source384_config(config_path)
        config_sha256 = _sha(_read(config_path, 131072))
        successor = predecessor is not None
        header_mode = config["schema"] == "terminal-source384-config@2" and not successor
        header_pin = _header_pin() if header_mode else None
        _require(header_mode == (intent_binding is not None), "header profile requires exact reviewed intent binding")
        if header_mode:
            from .header_intent_applicability import _contract_binding
            _require(type(intent_binding) is dict and set(intent_binding) == {"contract", "manifest_cid"},
                     "closed immutable intent binding required")
            _contract_binding(intent_binding["contract"], intent_binding["manifest_cid"], source_hashes, config)
        exclusions = [".runtime", *excluded_new_output_paths]
        head = index.prepare_current(root, repository_id=repository_id, operation_id="initial",
            expected_head=None, limits=limits, exclusions=exclusions, parent_lease=lease,
            timeout_seconds=remaining(), memory_mb=4096).head
        inventory = _inventory(index, head, source_hashes)
        paths = _program_python_paths(inventory, program_scope)
        _require(bool(paths), "Source384 abstained: no captured Python program inputs; checkpoint not consumed")
        version = register_shared_parent(registry, checkpoint_path=config["checkpoint_path"],
                                        expected_sha256=config["checkpoint_sha256"])
        inference = infer_shared_parent_units(index, root, expected_head=head, registry=registry,
            version_id=version, paths=paths, embedding_snapshot=config["embedding_snapshot"],
            parent_lease=lease, timeout_seconds=remaining(), memory_mb=4096)
        _validate_population(inference, inventory, program_scope)
        nomination = None
        if header_mode:
            from .header_intent_applicability import prepare_runtime_nomination
            nomination = prepare_runtime_nomination(index, head=head, output=output, config_path=config_path,
                config=config, intent_binding=intent_binding, source_hashes=source_hashes,
                parent_lease=lease, remaining=remaining)
        _sources(root, source_hashes)
        _require(config == load_source384_config(config_path) and producer == _pins()
                 and config_sha256 == _sha(_read(config_path, 131072)),
                 "Source384 selected assets or producer changed during preparation")
        inference_sha256 = _write(output / "inference.json", inference, MAX_INFERENCE_BYTES)
        remaining()
        receipt = dict(schema=SUCCESSOR_SCHEMA if successor else HEADER_SCHEMA if header_mode else SCHEMA,
            output=str(output), repository=str(root),
            source_hashes=source_hashes, source_head=head.to_dict(), source_inventory=inventory,
            config_path=str(config_path), config_sha256=config_sha256,
            checkpoint_sha256=config["checkpoint_sha256"], version_id=version,
            inference_sha256=inference_sha256, producer=producer,
            summary=_summary(inference, inventory=inventory,
                checkpoint_sha256=config["checkpoint_sha256"], inference_sha256=inference_sha256,
                advisory_successor=successor, program_scope=program_scope),
            seconds=time.monotonic() - started, training_steps=0, neural_inference_replayed=False,
            resource_profile=dict(parent_memory_mb=6144, parent_cpu_slots=3,
                numerical_worker_memory_mb=4096, memory_enforcement="sampled_process_tree_RSS_may_overshoot"),
            **AUTHORITY)
        if program_scope is not None:
            receipt["program_scope"] = program_scope
            _scope_selection(root, output, source_hashes, program_scope)
        if successor:
            _require(all(receipt[key] == predecessor[key] for key in
                ("producer", "config_path", "config_sha256", "checkpoint_sha256", "version_id")),
                "Source384 successor changed its selected producer/model/configuration")
            validate_historical_source384_selection(repository=root, expected_receipt=predecessor)
            _require(inference.get("native_worker_executed") is True
                     and inference.get("inference_executed") is True,
                     "Source384 successor requires fresh native inference")
            receipt.update(successor_lineage=_successor_lineage(predecessor, excluded_new_output_paths),
                           requires_independent_manifest=True, planning_authority=False, dispatch_authority=False,
                           inference_execution=dict(native_worker_executed=True, inference_executed=True,
                               model_loads=inference["report"]["output"]["model_loads"]))
            _validate_successor_envelope(receipt)
        if header_mode:
            receipt["source_applicability_nomination"] = nomination
            _require(header_pin == _header_pin(), "header consumer changed during preparation")
            receipt["header_consumer_sha256"] = header_pin
        _write(output / "receipt.json", receipt, MAX_RECEIPT_BYTES)
        remaining()
    return receipt


def validate_source384_context(*, repository, expected_receipt, scheduler=None,
                               parent_lease=None, timeout_seconds=90.):
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources, codebase_admission_timeout
    _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 180,
             "bounded Source384 observation deadline required")
    started = time.monotonic()
    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease,
            timeout_seconds=codebase_admission_timeout(remaining_seconds=max(0., started + timeout_seconds - time.monotonic())),
            memory_mb=6144, cpu_slots=3,
            child_process_slots=3) as lease:
        deadline = started + timeout_seconds
        _remaining(deadline)
        receipt = _validate_source384_context(repository=repository, expected_receipt=expected_receipt,
                                             parent_lease=lease, deadline=deadline)
        _remaining(deadline)
        return receipt


def _validate_source384_context(*, repository, expected_receipt, parent_lease, deadline):
    from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
    from ipfs_datasets_py.logic.software_contracts.codebase_source_units_384 import validate_shared_parent_units
    root = Path(repository).resolve(strict=True)
    _require(type(expected_receipt) is dict and expected_receipt.get("schema") in {SCHEMA, HEADER_SCHEMA, SUCCESSOR_SCHEMA},
             "selected Source384 context receipt required")
    output = _output(root, expected_receipt["output"])
    receipt = json.loads(_read(output / "receipt.json", MAX_RECEIPT_BYTES))
    _require(receipt == expected_receipt and receipt["repository"] == str(root)
             and receipt["producer"] == _pins() and receipt["training_steps"] == 0
             and all(receipt.get(key) is False for key in AUTHORITY),
             "Source384 receipt, producer or authority changed")
    source_hashes = _sources(root, receipt["source_hashes"])
    program_scope = receipt.get("program_scope")
    _scope_selection(root, output, source_hashes, program_scope)
    config_path = Path(receipt["config_path"])
    config = load_source384_config(config_path)
    header_mode = receipt["schema"] == HEADER_SCHEMA
    successor = receipt["schema"] == SUCCESSOR_SCHEMA
    if successor:
        _validate_successor_envelope(receipt)
    _require(not header_mode or receipt.get("header_consumer_sha256") == _header_pin(), "header consumer changed")
    _require((successor or header_mode == (config["schema"] == "terminal-source384-config@2"))
        and header_mode == ("source_applicability_nomination" in receipt), "Source384 header profile selection changed")
    if successor:
        lineage = receipt.get("successor_lineage", {})
        predecessor_output = _output(root, lineage["predecessor_output"])
        _require(predecessor_output != output, "Source384 successor cannot be its own predecessor")
        predecessor = json.loads(_read(predecessor_output / "receipt.json", MAX_RECEIPT_BYTES))
        validate_historical_source384_selection(repository=root, expected_receipt=predecessor)
        _require(predecessor.get("successor_lineage", {}).get("predecessor_output") != str(output),
                 "cyclic Source384 predecessor lineage")
        exclusions = _successor_exclusions(lineage.get("excluded_new_output_paths"), source_hashes)
        _require(lineage == _successor_lineage(predecessor, exclusions)
                 and set(source_hashes) == set(predecessor["source_hashes"])
                 and all(receipt[key] == predecessor[key] for key in
                    ("producer", "config_path", "config_sha256", "checkpoint_sha256", "version_id"))
                 and receipt.get("requires_independent_manifest") is True
                 and receipt.get("planning_authority") is False and receipt.get("dispatch_authority") is False
                 and "header_consumer_sha256" not in receipt,
                 "Source384 successor lineage or authority changed")
    _require(_sha(_read(config_path, 131072)) == receipt["config_sha256"]
             and config["checkpoint_sha256"] == receipt["checkpoint_sha256"],
             "Source384 selected configuration changed")
    raw = _read(output / "inference.json", MAX_INFERENCE_BYTES)
    _require(_sha(raw) == receipt["inference_sha256"], "Source384 inference digest differs")
    inference = json.loads(raw)
    del raw
    if successor:
        _require(inference.get("native_worker_executed") is True
                 and inference.get("inference_executed") is True
                 and receipt.get("inference_execution") == dict(native_worker_executed=True,
                     inference_executed=True, model_loads=inference["report"]["output"]["model_loads"]),
                 "Source384 successor inference observation changed")
    key = inference["report"]["key"]
    _require(key["source_head"] == receipt["source_head"]
             and key["version_id"] == receipt["version_id"]
             and key["original_checkpoint_sha256"] == receipt["checkpoint_sha256"],
             "Source384 inference differs from selected source/model")
    with _owners(output) as (index, registry):
        head = CodebaseHead.from_dict(receipt["source_head"])
        inventory = _inventory(index, head, source_hashes)
        _require(inventory == receipt["source_inventory"], "Source384 inventory differs")
        _validate_population(inference, inventory, program_scope)
        validate_shared_parent_units(index, root, inference, registry=registry,
            embedding_snapshot=config["embedding_snapshot"],
            parent_lease=parent_lease, timeout_seconds=_remaining(deadline), memory_mb=4096)
        if header_mode:
            from .header_intent_applicability import validate_captured_nomination
            validate_captured_nomination(index, head=head, nomination=receipt["source_applicability_nomination"],
                config=config, config_path=config_path, output=output, source_hashes=source_hashes,
                parent_lease=parent_lease, remaining=lambda: _remaining(deadline))
    _require(receipt["summary"] == _summary(inference, inventory=inventory,
        checkpoint_sha256=receipt["checkpoint_sha256"], inference_sha256=receipt["inference_sha256"],
        advisory_successor=successor, program_scope=program_scope),
        "Source384 summary does not match native coverage")
    _sources(root, source_hashes)
    _scope_selection(root, output, source_hashes, program_scope)
    _require(config == load_source384_config(config_path) and receipt["producer"] == _pins()
             and _sha(_read(config_path, 131072)) == receipt["config_sha256"]
             and json.loads(_read(output / "receipt.json", MAX_RECEIPT_BYTES)) == receipt
             and _sha(_read(output / "inference.json", MAX_INFERENCE_BYTES)) == receipt["inference_sha256"],
             "Source384 context changed during observation")
    _require(not header_mode or receipt.get("header_consumer_sha256") == _header_pin(), "header consumer changed during replay")
    _remaining(deadline)
    return receipt


def _header_pin():
    from . import header_intent_applicability
    return _sha(Path(header_intent_applicability.__file__).read_bytes())
