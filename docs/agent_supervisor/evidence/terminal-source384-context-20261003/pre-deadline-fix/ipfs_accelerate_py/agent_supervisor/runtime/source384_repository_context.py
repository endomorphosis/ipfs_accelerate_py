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

SCHEMA = "terminal-source384-repository-context@1"
MAX_RECEIPT_BYTES = 131072
MAX_INFERENCE_BYTES = 32 * 1024 * 1024
AUTHORITY = dict(proof_authority=False, execution_authority=False,
                 completion_authority=False, formalization_authority=False)


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
    from . import source384_config, security_autoencoder_advisor
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
                               scanner, snapshot, python_analysis, content, python_frontend)}}


def _write(path, value, maximum):
    raw = _raw(value)
    _require(len(raw) <= maximum, "Source384 artifact exceeds its byte bound")
    with path.open("xb") as stream:
        stream.write(raw)
    return _sha(raw)


def _sources(repository, source_hashes):
    _require(type(source_hashes) is dict and 1 <= len(source_hashes) <= 128,
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


def _summary(inference, *, inventory, checkpoint_sha256, inference_sha256):
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
    _require(len(_raw(result)) <= 8192, "Source384 planning summary exceeds fixed bound")
    return result


def _validate_population(inference, inventory):
    preparation = inference["report"]["preparation"]
    paths = sorted(row["path"] for row in inventory if row["disposition"] == "captured"
                   and row["path"].endswith(".py"))
    _require(preparation["paths"] == paths and preparation["max_functions"] == 1024
             and preparation["max_selected_units"] == 128,
             "Source384 inference selection differs from the complete declared Python population")


def prepare_source384_context(*, repository, source_hashes, output, config_path,
                              timeout_seconds=90., scheduler=None, parent_lease=None):
    from ipfs_datasets_py.logic.software_contracts.codebase_ir import CodebaseScanLimits
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.logic.software_contracts.codebase_source_384 import register_shared_parent
    from ipfs_datasets_py.logic.software_contracts.codebase_source_units_384 import infer_shared_parent_units
    root = Path(repository).resolve(strict=True)
    output = _output(root, output, fresh=True)
    config_path = Path(config_path)
    source_hashes = _sources(root, source_hashes)
    producer = _pins()
    started = time.monotonic()
    _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 180,
             "bounded Source384 preparation deadline required")
    def remaining():
        left = timeout_seconds - (time.monotonic() - started)
        _require(left > 0, "Source384 preparation deadline expired")
        return left
    output.mkdir(mode=0o700)
    # Bounds cover signed input only; the native scanner refuses larger trees.
    limits = CodebaseScanLimits(max_entries=len(source_hashes), max_file_bytes=1024 * 1024)
    repository_id = "terminal-source384:" + _sha(_raw(dict(repository=str(root), sources=source_hashes)))
    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease,
            timeout_seconds=min(30., remaining()), memory_mb=6144, cpu_slots=3,
            child_process_slots=3) as lease, _owners(output) as (index, registry):
        config = load_source384_config(config_path)
        config_sha256 = _sha(_read(config_path, 131072))
        head = index.prepare_current(root, repository_id=repository_id, operation_id="initial",
            expected_head=None, limits=limits, exclusions=[".runtime"], parent_lease=lease,
            timeout_seconds=remaining(), memory_mb=4096).head
        inventory = _inventory(index, head, source_hashes)
        paths = [row["path"] for row in inventory if row["disposition"] == "captured"
                 and row["path"].endswith(".py")]
        _require(bool(paths), "Source384 profile requires captured Python source")
        version = register_shared_parent(registry, checkpoint_path=config["checkpoint_path"],
                                        expected_sha256=config["checkpoint_sha256"])
        inference = infer_shared_parent_units(index, root, expected_head=head, registry=registry,
            version_id=version, paths=paths, embedding_snapshot=config["embedding_snapshot"],
            parent_lease=lease, timeout_seconds=remaining(), memory_mb=4096)
        _validate_population(inference, inventory)
        _sources(root, source_hashes)
        _require(config == load_source384_config(config_path) and producer == _pins()
                 and config_sha256 == _sha(_read(config_path, 131072)),
                 "Source384 selected assets or producer changed during preparation")
        inference_sha256 = _write(output / "inference.json", inference, MAX_INFERENCE_BYTES)
        remaining()
        receipt = dict(schema=SCHEMA, output=str(output), repository=str(root),
            source_hashes=source_hashes, source_head=head.to_dict(), source_inventory=inventory,
            config_path=str(config_path), config_sha256=config_sha256,
            checkpoint_sha256=config["checkpoint_sha256"], version_id=version,
            inference_sha256=inference_sha256, producer=producer,
            summary=_summary(inference, inventory=inventory,
                checkpoint_sha256=config["checkpoint_sha256"], inference_sha256=inference_sha256),
            seconds=time.monotonic() - started, training_steps=0, neural_inference_replayed=False,
            resource_profile=dict(parent_memory_mb=6144, parent_cpu_slots=3,
                numerical_worker_memory_mb=4096, memory_enforcement="sampled_process_tree_RSS_may_overshoot"),
            **AUTHORITY)
        _write(output / "receipt.json", receipt, MAX_RECEIPT_BYTES)
        remaining()
    return receipt


def validate_source384_context(*, repository, expected_receipt, scheduler=None,
                               parent_lease=None, timeout_seconds=90.):
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    _require(type(timeout_seconds) in (int, float) and 0 < timeout_seconds <= 180,
             "bounded Source384 observation deadline required")
    started = time.monotonic()
    with acquire_codebase_resources(scheduler=scheduler, parent_lease=parent_lease,
            timeout_seconds=min(30., timeout_seconds), memory_mb=6144, cpu_slots=3,
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
    _require(type(expected_receipt) is dict and expected_receipt.get("schema") == SCHEMA,
             "selected Source384 context receipt required")
    output = _output(root, expected_receipt["output"])
    receipt = json.loads(_read(output / "receipt.json", MAX_RECEIPT_BYTES))
    _require(receipt == expected_receipt and receipt["repository"] == str(root)
             and receipt["producer"] == _pins() and receipt["training_steps"] == 0
             and all(receipt.get(key) is False for key in AUTHORITY),
             "Source384 receipt, producer or authority changed")
    source_hashes = _sources(root, receipt["source_hashes"])
    config_path = Path(receipt["config_path"])
    config = load_source384_config(config_path)
    _require(_sha(_read(config_path, 131072)) == receipt["config_sha256"]
             and config["checkpoint_sha256"] == receipt["checkpoint_sha256"],
             "Source384 selected configuration changed")
    raw = _read(output / "inference.json", MAX_INFERENCE_BYTES)
    _require(_sha(raw) == receipt["inference_sha256"], "Source384 inference digest differs")
    inference = json.loads(raw)
    key = inference["report"]["key"]
    _require(key["source_head"] == receipt["source_head"]
             and key["version_id"] == receipt["version_id"]
             and key["original_checkpoint_sha256"] == receipt["checkpoint_sha256"],
             "Source384 inference differs from selected source/model")
    with _owners(output) as (index, registry):
        head = CodebaseHead.from_dict(receipt["source_head"])
        inventory = _inventory(index, head, source_hashes)
        _require(inventory == receipt["source_inventory"], "Source384 inventory differs")
        _validate_population(inference, inventory)
        validate_shared_parent_units(index, root, inference, registry=registry,
            embedding_snapshot=config["embedding_snapshot"],
            parent_lease=parent_lease, timeout_seconds=_remaining(deadline), memory_mb=4096)
    _require(receipt["summary"] == _summary(inference, inventory=inventory,
        checkpoint_sha256=receipt["checkpoint_sha256"], inference_sha256=receipt["inference_sha256"]),
        "Source384 summary does not match native coverage")
    _sources(root, source_hashes)
    _require(config == load_source384_config(config_path) and receipt["producer"] == _pins()
             and _sha(_read(config_path, 131072)) == receipt["config_sha256"]
             and json.loads(_read(output / "receipt.json", MAX_RECEIPT_BYTES)) == receipt
             and _sha(_read(output / "inference.json", MAX_INFERENCE_BYTES)) == receipt["inference_sha256"],
             "Source384 context changed during observation")
    _remaining(deadline)
    return receipt
