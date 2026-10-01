"""Reusable offline frozen Security-CVE advice with native world/index receipts.

This local preparation service emits candidate advice only. It never trains,
downloads, changes tasks, admits repairs, or treats a classifier as a prover.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import stat

from ..proof.formal_verification_contracts import content_identity
from ..semantic_state.program_world_database import ProgramWorldDatabase
from .supervisor_meta_index import SupervisorMetaIndex

SCHEMA = "supervisor-frozen-security-advice@1"
OPERATION = "frozen_security_inference_observation"
AUTHORITY = {"authority": "unverified_candidate_only", "completion_authority": False,
             "proof_authority": False, "formalization_authority": False,
             "execution_authority": False, "omission_authority": False}


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, maximum=8_000_000):
    path = Path(path)
    if path.resolve(strict=True) != path or not path.is_absolute():
        raise ValueError("canonical security advice artifact required")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > maximum:
            raise ValueError("bounded regular security advice artifact required")
        raw = stream.read(maximum + 1)
        after = os.fstat(stream.fileno())
    identity = lambda value: (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns)
    if (len(raw) > maximum or identity(after) != identity(info)
            or identity(path.lstat()) != identity(info) or path.resolve(strict=True) != path):
        raise ValueError("security advice artifact changed during read")
    return raw


def _write(path, value):
    raw = _bytes(value)
    with Path(path).open("xb") as stream:
        stream.write(raw)
    return _sha(raw)


def _output(repository, output, *, fresh, checkpoint):
    path = Path(output)
    package = Path(checkpoint["output"])
    if (not path.is_absolute() or path.resolve() != path or path.is_relative_to(repository)
            or repository.is_relative_to(path) or path.is_relative_to(package) or package.is_relative_to(path)
            or (fresh and path.exists())):
        raise ValueError("fresh canonical external security advice state required")
    return path


def _source_hashes(repository, paths):
    if not paths or len(paths) != len(set(paths)) or len(paths) > 64:
        raise ValueError("bounded explicit security source paths required")
    result = {}
    for name in sorted(paths):
        relative = Path(name)
        if (relative.is_absolute() or relative.as_posix() != name or ".." in relative.parts
                or not name.endswith(".py")):
            raise ValueError("canonical declared Python security source required")
        result[name] = _sha(_read(repository / name, 4_000_000))
    return result


def _summary(inference, formalization=None):
    nominations = inference.get("security_candidate_nominations") or {}
    ranks = [{key: row[key] for key in ("row_id", "path", "symbol", "line", "reconstruction_error")}
             for row in inference["ranks"][:5]]
    selected = {row["row_id"] for row in ranks}
    summary = {"schema": "frozen-security-planning-summary@1", "mode": "frozen_inference",
        "checkpoint_sha256": inference["checkpoint"]["checkpoint_sha256"],
        "manifest_sha256": inference["checkpoint"]["manifest_sha256"],
        "inference_sha256": inference["inference_sha256"],
        "source_hashes": inference["source_hashes"], "sample_count": inference["sample_count"],
        "ranked_candidates": ranks, "omitted_candidates": inference["sample_count"] - len(ranks),
        "target_vocabulary": nominations.get("target_vocabulary", []),
        "candidate_scores": [row for row in nominations.get("rows", []) if row["row_id"] in selected],
        "unsupported": inference.get("unsupported", []),
        "model_scope": "benchmark_informed_development", "unseen_target_classes_supported": False,
        "scores_are_calibrated_probabilities": False, "granularity_shift_validated": False,
        "formal_formula_heads_present": False, "holdout_evaluated": False,
        "training_steps": 0, "provider_calls": 0, "download_calls": 0, **AUTHORITY}
    if formalization is not None:
        summary["formalization"] = {"report_cid": formalization["report_cid"],
            **formalization["summary"], "producer": "datasets_security_formula_decoder"}
        summary["formal_formula_heads_present"] = True
    if len(_bytes(summary)) > 8192:
        raise ValueError("frozen security advice exceeds fixed planner summary bound")
    return summary


def _catalogs(output, record, world_cid):
    catalogs = [("metadata", record["package"], "sha256:" + record["manifest_sha256"]),
            ("vector", record["inference_output"], "sha256:" + record["inference_sha256"]),
            ("world_model", str(output / "world.duckdb"), world_cid)]
    if "formalization" in record:
        catalogs.extend([
            ("metadata", record["formula_decoder"]["output"], content_identity(record["formula_decoder"])),
            ("ast", record["formalization"]["output"], record["formalization"]["report_cid"]),
            ("knowledge_graph", str(output / "formalization-plan.json"),
             "sha256:" + record["formalization_plan_sha256"]),
        ])
    return catalogs


def _inventory(output):
    """Bind every persisted observation/catalog byte, including lake payloads."""
    files, total = {}, 0
    for path in sorted(output.rglob("*")):
        if path == output / "receipt.json":
            continue
        info = path.lstat()
        if stat.S_ISDIR(info.st_mode):
            continue
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("security advice inventory contains a link or special file")
        raw = _read(path, 16_000_000)
        total += len(raw)
        if total > 64_000_000 or len(files) >= 128:
            raise ValueError("security advice artifact inventory exceeds bounds")
        files[path.relative_to(output).as_posix()] = {"sha256": _sha(raw), "bytes": len(raw)}
    for lake in ("world-lake", "metadata-lake"):
        if lake + "/metadata.ducklake" not in files or not any(
                name.startswith(lake + "/parquet/") and name.endswith(".parquet") for name in files):
            raise ValueError("security advice DuckLake artifact inventory is incomplete")
    return files


def _hydrate_lakes(output, record, world_cid):
    """Read native projected rows without granting the lakes task authority."""
    import duckdb
    with duckdb.connect(str(output / "metadata.duckdb"), read_only=True, config={"threads": 1}) as primary:
        catalogs = primary.execute("SELECT catalog_cid, kind, locator_ref, exclusive_owner, "
            "attach_permitted, tree_id, recorded_at FROM catalogs ORDER BY catalog_cid").fetchall()
        links = primary.execute("SELECT link_cid, subject_kind, subject_ref, catalog_id, "
            "record_kind, record_ref, freshness_mtime_ns, recorded_at FROM identity_links ORDER BY link_cid").fetchall()
    with duckdb.connect(":memory:", config={"threads": 1, "memory_limit": "256MB"}) as lake:
        lake.execute("SET autoinstall_known_extensions=false")
        lake.execute("SET autoload_known_extensions=false")
        lake.execute("LOAD ducklake")
        for directory, alias in (("world-lake", "security_world"), ("metadata-lake", "security_metadata")):
            catalog = str(output / directory / "metadata.ducklake").replace("'", "''")
            data = str(output / directory / "parquet").replace("'", "''")
            lake.execute("ATTACH 'ducklake:" + catalog + "' AS " + alias + " (READ_ONLY, DATA_PATH '" + data + "')")
        world_rows = lake.execute("SELECT record_cid, payload_json, completion_authority "
            "FROM security_world.program_world_records").fetchall()
        if len(world_rows) != 1 or world_rows[0][0] != world_cid or json.loads(world_rows[0][1]) != record or world_rows[0][2] is not False:
            raise ValueError("security advice DuckLake world projection differs")
        projected_catalogs = lake.execute("SELECT catalog_cid, kind, locator_ref, exclusive_owner, "
            "attach_permitted, tree_id, recorded_at FROM security_metadata.catalogs ORDER BY catalog_cid").fetchall()
        projected_links = lake.execute("SELECT link_cid, subject_kind, subject_ref, catalog_id, "
            "record_kind, record_ref, freshness_mtime_ns, recorded_at FROM security_metadata.identity_links ORDER BY link_cid").fetchall()
        authority = lake.execute("SELECT count(*) FROM security_metadata.catalogs WHERE completion_authority IS DISTINCT FROM FALSE").fetchone()[0]
        authority += lake.execute("SELECT count(*) FROM security_metadata.identity_links WHERE completion_authority IS DISTINCT FROM FALSE").fetchone()[0]
        if projected_catalogs != catalogs or projected_links != links or authority:
            raise ValueError("security advice DuckLake metadata projection differs")
    return {"ducklake_verified": True, "ducklake_world_records": 1,
            "ducklake_catalogs": len(catalogs), "ducklake_links": len(links)}


def _hydrate(output, record):
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    found = world.records_for_decision(task_id=record["task_id"], operation=OPERATION)
    if found["n"] != 1 or found["records"][0]["payload"] != record:
        raise ValueError("frozen security world observation differs")
    world_cid = found["records"][0]["record_cid"]
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    linked = meta.compose_for_subject(subject_kind="tree_id", subject_ref=record["source_tree_id"])
    expected = set(_catalogs(output, record, world_cid))
    if linked["n"] != len(expected) or {(row["catalog_kind"], row["locator_ref"], row["record_ref"])
            for row in linked["linked"]} != expected:
        raise ValueError("frozen security model/inference metadata differs")
    return {"world_record_cid": world_cid, "catalog_count": len(expected), "linked_count": linked["n"],
            **_hydrate_lakes(output, record, world_cid)}


def prepare_security_advice(*, repository: Path, paths: list[str], source_hashes: dict,
                            checkpoint: dict, output: Path, hub_descriptor: dict | None = None,
                            model_manager=None, formula_decoder: dict | None = None,
                            header_protocol: dict | None = None) -> dict:
    """Prepare offline inference for a local runtime's exact declared scope."""
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import infer_security_checkpoint, validate_security_inference
    repository = Path(repository).resolve(strict=True)
    output = _output(repository, output, fresh=True, checkpoint=checkpoint)
    if _source_hashes(repository, paths) != source_hashes:
        raise ValueError("security advice source differs from independent declaration")
    if header_protocol is not None and formula_decoder is None:
        raise ValueError("header protocol belongs to the explicit formalization selection")
    output.mkdir(parents=True, mode=0o700)
    from .security_autoencoder_hub import register_security_checkpoint
    if model_manager is None:
        from ...model_manager import ModelManager
        model_manager = ModelManager(storage_path=str(output / "model-registry.json"),
            use_database=False, enable_ipfs=False, project_legacy_models=False)
    registration = register_security_checkpoint(manager=model_manager, package=Path(checkpoint["output"]),
        expected_manifest_sha256=checkpoint["manifest_sha256"], hub_descriptor=hub_descriptor)
    inference = infer_security_checkpoint(repository=repository, paths=paths, source_hashes=source_hashes,
        checkpoint=checkpoint, output=output / "inference")
    validate_security_inference(repository=repository, expected_receipt=inference)
    inference_descriptor_sha256 = _write(output / "inference-descriptor.json", inference)
    formalization = plan = None
    if formula_decoder is not None:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formalization_pipeline import (
            run_security_formalization_pipeline, validate_security_formalization_pipeline,
        )
        from .security_formalization_plan import compile_security_formalization_plan
        from .security_formula_model import register_security_formula_decoder
        formula_registration = register_security_formula_decoder(manager=model_manager, checkpoint=formula_decoder)
        formalization = run_security_formalization_pipeline(repository=repository,
            source_hashes=source_hashes, decoder=formula_decoder, protocol=header_protocol,
            output=output / "formalization")
        report = validate_security_formalization_pipeline(repository=repository, receipt=formalization)
        plan = compile_security_formalization_plan(report=report)
        _write(output / "formalization-plan.json", plan)
    summary = _summary(inference, formalization)
    tree = content_identity({"schema": "frozen-security-source@1", "sources": source_hashes})
    record = {"task_id": "security-observation:" + inference["inference_sha256"],
        "board": "frozen-security-advice", "operation": OPERATION, "repository": str(repository),
        "source_tree_id": tree, "source_hashes": source_hashes, "package": checkpoint["output"],
        "manifest_sha256": checkpoint["manifest_sha256"], "checkpoint_sha256": checkpoint["checkpoint_sha256"],
        "inference_output": inference["output"], "inference_sha256": inference["inference_sha256"],
        "inference_descriptor_sha256": inference_descriptor_sha256,
        "sample_count": inference["sample_count"], "summary_sha256": _sha(_bytes(summary)),
        "model_id": registration["model_id"], "model_catalog_revision": registration["catalog_revision"],
        "model_registration_sha256": _sha(_bytes(registration)),
        "target_vocabulary": summary["target_vocabulary"], "training_steps": 0,
        "provider_calls": 0, "download_calls": 0, "proposal_only": True,
        "cas_completed": False, "admitted": False, **AUTHORITY}
    if formalization is not None:
        record.update(formalization=formalization, formula_decoder=formula_decoder,
            header_protocol=header_protocol, formalization_plan_sha256=_sha(_bytes(plan)),
            formula_registration=formula_registration)
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    persisted = world.persist(record)
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    for kind, locator, ref in _catalogs(output, record, persisted["record_cid"]):
        catalog = meta.register_catalog(kind=kind, locator_ref=locator,
            repository_id=content_identity({"repository": str(repository)}), tree_id=tree, project=False)
        meta.link_identity(subject_kind="tree_id", subject_ref=tree, catalog_id=catalog["catalog_id"],
            record_kind="frozen_security_" + kind, record_ref=ref, project=False)
    projected = meta.project_ducklake()
    if persisted["ducklake"]["status"] != "projected" or projected["status"] != "projected":
        raise ValueError("frozen security native DuckLake projection unavailable")
    receipt = {"schema": SCHEMA, "status": "current", "mode": "frozen_inference",
        "repository": str(repository), "output": str(output), "checkpoint": checkpoint,
        "source_hashes": source_hashes, "paths": sorted(paths), "record": record,
        "summary": summary, "hydration": _hydrate(output, record),
        "model_registration": registration,
        "world_ducklake": persisted["ducklake"], "metadata_ducklake": projected,
        "files": _inventory(output),
        "training_steps": 0, "provider_calls": 0, "download_calls": 0, **AUTHORITY}
    if formalization is not None:
        receipt.update(formalization=formalization, formula_decoder=formula_decoder,
                       header_protocol=header_protocol, formula_registration=formula_registration)
    validate_security_inference(repository=repository, expected_receipt=inference)
    return {**receipt, "receipt_sha256": _write(output / "receipt.json", receipt)}


def _load(repository, expected_receipt, *, replay_formal_source=True):
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import load_security_checkpoint
    if (type(expected_receipt) is not dict or expected_receipt.get("schema") != SCHEMA
            or expected_receipt.get("repository") != str(repository)
            or expected_receipt.get("mode") != "frozen_inference"
            or any(expected_receipt.get(key) != value for key, value in AUTHORITY.items())
            or any(type(expected_receipt.get(key)) is not int or expected_receipt[key] != 0
                   for key in ("training_steps", "provider_calls", "download_calls"))):
        raise ValueError("exact frozen security advice receipt required")
    output = _output(repository, expected_receipt["output"], fresh=False, checkpoint=expected_receipt["checkpoint"])
    raw = _read(output / "receipt.json")
    if (_sha(raw) != expected_receipt.get("receipt_sha256")
            or json.loads(raw) != {key: value for key, value in expected_receipt.items() if key != "receipt_sha256"}):
        raise ValueError("frozen security advice receipt changed")
    if _inventory(output) != expected_receipt.get("files"):
        raise ValueError("frozen security advice artifact inventory changed")
    checkpoint = expected_receipt["checkpoint"]
    checked = load_security_checkpoint(Path(checkpoint["output"]), expected_manifest_sha256=checkpoint["manifest_sha256"])
    if checked["descriptor"] != checkpoint:
        raise ValueError("frozen security model differs from selected checkpoint")
    registration = expected_receipt["model_registration"]
    if (registration["checkpoint"] != checkpoint or registration["operation"] != "security.advise"
            or registration["inference_probe"].get("executed") is not True
            or _sha(_bytes(registration)) != expected_receipt["record"]["model_registration_sha256"]):
        raise ValueError("frozen security model registration binding differs")
    raw = _read(output / "inference-descriptor.json")
    if _sha(raw) != expected_receipt["record"]["inference_descriptor_sha256"]:
        raise ValueError("frozen security inference descriptor changed")
    inference = json.loads(raw)
    formalization = expected_receipt.get("formalization")
    if formalization is not None:
        from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formalization_pipeline import validate_security_formalization_pipeline
        from .security_formalization_plan import compile_security_formalization_plan
        from .security_formula_model import INFERENCE_CODE, _probe
        if (formalization["output"] != str(output / "formalization")
                or expected_receipt["record"].get("formalization") != formalization
                or expected_receipt["record"].get("formula_registration") != expected_receipt.get("formula_registration")
                or expected_receipt["formula_registration"]["checkpoint"] != expected_receipt["formula_decoder"]):
            raise ValueError("formalization observation differs from persisted world record")
        registered = expected_receipt["formula_registration"]
        if (registered.get("operation") != "security.advise"
                or registered.get("inference_code") != INFERENCE_CODE
                or registered.get("inference_probe") != _probe(expected_receipt["formula_decoder"])
                or registered.get("text_generation") is not False
                or registered.get("proof_authority") is not False):
            raise ValueError("formula model registration capability differs")
        if replay_formal_source:
            report = validate_security_formalization_pipeline(repository=repository, receipt=formalization)
        else:
            # Refresh starts from an intact historical observation. Current
            # source replay happens after re-inference into the fresh output.
            from ipfs_datasets_py.logic.formalization.autoencoder.security.security_formula_decoder import load_security_formula_decoder
            load_security_formula_decoder(expected_receipt["formula_decoder"])
            historical = _read(output / "formalization/formalization.json", 32 * 1024 * 1024)
            if _sha(historical) != formalization["report_sha256"]:
                raise ValueError("historical formalization observation changed")
            report = json.loads(historical)
        if (report["source_hashes"] != expected_receipt["source_hashes"]
                or report["decoder"] != expected_receipt["formula_decoder"]
                or report["protocol"] != expected_receipt["header_protocol"]
                or report["model_enabled"] is not True
                or any(expected_receipt["record"].get(key) != expected_receipt[key]
                       for key in ("formula_decoder", "header_protocol"))):
            raise ValueError("formalization model, source or protocol selection differs")
        plan = compile_security_formalization_plan(report=report)
        raw_plan = _read(output / "formalization-plan.json", 16_000_000)
        if (raw_plan != _bytes(plan)
                or _sha(raw_plan) != expected_receipt["record"]["formalization_plan_sha256"]):
            raise ValueError("native formalization plan changed")
    elif any(key in expected_receipt or key in expected_receipt["record"]
             for key in ("formula_decoder", "header_protocol", "formalization")):
        raise ValueError("selected formula decoder evidence is missing")
    if (inference["checkpoint"] != checkpoint or inference["source_hashes"] != expected_receipt["source_hashes"]
            or inference["output"] != str(output / "inference")
            or _summary(inference, formalization) != expected_receipt["summary"]
            or _hydrate(output, expected_receipt["record"]) != expected_receipt["hydration"]):
        raise ValueError("frozen security inference/model/world binding differs")
    if _inventory(output) != expected_receipt["files"]:
        raise ValueError("frozen security advice artifacts changed during hydration")
    return inference


def validate_security_advice(*, repository: Path, expected_receipt: dict) -> dict:
    from ipfs_datasets_py.logic.formalization.autoencoder.security.security_autoencoder_checkpoint import validate_security_inference
    repository = Path(repository).resolve(strict=True)
    inference = _load(repository, expected_receipt)
    if _source_hashes(repository, expected_receipt["paths"]) != expected_receipt["source_hashes"]:
        raise ValueError("frozen security advice is stale for current source")
    validate_security_inference(repository=repository, expected_receipt=inference)
    return expected_receipt["summary"]


def refresh_security_advice(*, repository: Path, previous: dict, output: Path) -> dict:
    """Recompute changed-source observations with identical frozen weights."""
    repository = Path(repository).resolve(strict=True)
    _load(repository, previous, replay_formal_source=False)
    hashes = _source_hashes(repository, previous["paths"])
    if hashes == previous["source_hashes"]:
        validate_security_advice(repository=repository, expected_receipt=previous)
        return previous
    return prepare_security_advice(repository=repository, paths=previous["paths"], source_hashes=hashes,
        checkpoint=previous["checkpoint"], output=output, hub_descriptor=previous["model_registration"]["hub"],
        formula_decoder=previous.get("formula_decoder"), header_protocol=previous.get("header_protocol"))
