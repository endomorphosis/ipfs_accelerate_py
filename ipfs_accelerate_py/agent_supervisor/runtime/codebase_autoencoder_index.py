"""Native world/metadata/DuckLake projection of a verified code AE index.

Numeric model artifacts remain inert, byte-hashed blobs. The native catalogs
hold source/checkpoint references and ranking order, never neural proof or
formalization authority. The original training scope remains explicit.
"""
from __future__ import annotations

import json
from pathlib import Path
import stat

from ..proof.formal_verification_contracts import content_identity
from ..semantic_state.program_world_database import ProgramWorldDatabase
from .supervisor_meta_index import SupervisorMetaIndex
from ipfs_datasets_py.logic.formalization.autoencoder.security.codebase_autoencoder import DOMAIN, _json, _read, _sha, _write, validate_codebase_autoencoder

SCHEMA = "supervisor-code-autoencoder-index@1"
OPERATION = "code_autoencoder_observation"


def _output(repository, learner, output, *, fresh):
    output = Path(output)
    if (not output.is_absolute() or output.resolve() != output
            or output.is_relative_to(repository) or output.is_relative_to(Path(learner["output"]))
            or Path(learner["output"]).is_relative_to(output)
            or (fresh and output.exists())
            or any(part.lower() in {"legal-ir", "legal_ir", "legalir", "shared-weights", "shared_weights"} for part in output.parts)):
        raise ValueError("fresh external code autoencoder index namespace required")
    return output


def _record(repository, learner, checked):
    descriptor_sha = _sha(_json(learner))
    tree = content_identity({"schema": "code-autoencoder-source-snapshot@1", "sources": learner["source_hashes"]})
    artifacts = {}
    for name in ("checkpoint", "features", "index", "receipt"):
        path = Path(learner["output"]) / (name + ".json")
        raw = _read(path)
        artifacts[name] = {"path": str(path), "sha256": _sha(raw), "bytes": len(raw)}
    # The native identity encodes this exact ranking order. Float scores/latents
    # are bound through index.json's byte hash and freshly replayed by learner.
    ranks = [{"rank": rank, **{key: row[key] for key in ("row_id", "path", "symbol", "line")}}
             for rank, row in enumerate(checked["ranks"], start=1)]
    record = {"task_id": "code-index:" + learner["checkpoint_sha256"], "board": "code-autoencoder",
        "operation": OPERATION, "domain": DOMAIN, "repository": str(repository), "source_tree_id": tree,
        "source_hashes": learner["source_hashes"], "learner_descriptor_sha256": descriptor_sha,
        "learner_receipt_sha256": learner["receipt_sha256"], "checkpoint_sha256": learner["checkpoint_sha256"],
        "source_count": learner["source_count"], "sample_count": learner["sample_count"],
        "artifacts": artifacts, "ranks": ranks, "training_scope": learner["metrics"]["training_scope"],
        "holdout_evaluated": False, "authority": "unverified_candidate_only", "proposal_only": True,
        "completion_authority": False, "formalization_authority": False, "proof_authority": False,
        "omission_authority": False, "cas_completed": False, "admitted": False}
    nominations = learner.get("security_candidate_nominations")
    if nominations is not None:
        record["security_candidate_nominations"] = {"schema": nominations["schema"],
            "row_count": len(nominations["rows"]), "target_vocabulary": nominations["target_vocabulary"],
            "nomination_bytes_sha256": _sha(_json(nominations)),
            "index_artifact_sha256": artifacts["index"]["sha256"],
            "canonical_export_manifest_sha256": nominations["canonical_export_manifest_sha256"],
            "authority": "unverified_candidate_only", "proof_authority": False,
            "formalization_authority": False, "execution_authority": False}
    return record


def _hydrate(output, record):
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    hydrated = world.records_for_decision(task_id=record["task_id"], operation=OPERATION)
    if hydrated["n"] != 1 or hydrated["records"][0]["payload"] != record:
        raise ValueError("native code autoencoder world hydration differs")
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    links = meta.compose_for_subject(subject_kind="tree_id", subject_ref=record["source_tree_id"])
    expected = {("world_model", str(output / "world.duckdb"), hydrated["records"][0]["record_cid"])}
    for name in ("checkpoint", "features", "index"):
        artifact = record["artifacts"][name]
        expected.add(("vector" if name == "index" else "metadata", artifact["path"], "sha256:" + artifact["sha256"]))
    observed = {(row["catalog_kind"], row["locator_ref"], row["record_ref"]) for row in links["linked"]}
    if links["n"] != len(expected) or observed != expected:
        raise ValueError("native code autoencoder metadata hydration differs")
    return {"world_record_cid": hydrated["records"][0]["record_cid"], "catalog_count": len(expected),
            "linked_count": links["n"], "world_payload_sha256": _sha(_json(record))}


def _inventory(output):
    rows, total = {}, 0
    for path in sorted(output.rglob("*")):
        if path == output / "receipt.json":
            continue
        info = path.lstat()
        if stat.S_ISLNK(info.st_mode):
            raise ValueError("autoencoder catalog artifact must not be symlinked")
        if path.is_file():
            raw = _read(path, limit=16_000_000)
            total += len(raw)
            if total > 64_000_000 or len(rows) >= 128:
                raise ValueError("autoencoder catalog inventory bound exceeded")
            rows[path.relative_to(output).as_posix()] = {"sha256": _sha(raw), "bytes": len(raw)}
    return rows


def build_codebase_autoencoder_index(*, repository: Path, learner: dict, output: Path) -> dict:
    repository = Path(repository).absolute()
    output = _output(repository, learner, output, fresh=True)
    checked = validate_codebase_autoencoder(repository=repository, expected_receipt=learner)
    record = _record(repository, learner, checked)
    output.mkdir(parents=True, mode=0o700)
    world = ProgramWorldDatabase(output / "world.duckdb", output / "world-lake")
    persisted = world.persist(record)
    if persisted["ducklake"]["status"] != "projected":
        raise ValueError("native code autoencoder world DuckLake projection unavailable")
    meta = SupervisorMetaIndex(output / "metadata.duckdb", output / "metadata-lake")
    catalogs = [("world_model", str(output / "world.duckdb"), persisted["record_cid"])]
    catalogs += [("vector" if name == "index" else "metadata", record["artifacts"][name]["path"],
                  "sha256:" + record["artifacts"][name]["sha256"]) for name in ("checkpoint", "features", "index")]
    repository_id = content_identity({"repository": str(repository)})
    for kind, locator, reference in catalogs:
        catalog = meta.register_catalog(kind=kind, locator_ref=locator, repository_id=repository_id,
                                        tree_id=record["source_tree_id"], project=False)
        meta.link_identity(subject_kind="tree_id", subject_ref=record["source_tree_id"],
            catalog_id=catalog["catalog_id"], record_kind="code_autoencoder_" + kind,
            record_ref=reference, project=False)
    projection = meta.project_ducklake()
    if projection["status"] != "projected":
        raise ValueError("native code autoencoder metadata DuckLake projection unavailable")
    hydrated = _hydrate(output, record)
    validate_codebase_autoencoder(repository=repository, expected_receipt=learner)
    receipt = {"schema": SCHEMA, "status": "hydrated", "domain": DOMAIN, "repository": str(repository), "output": str(output),
        "index_implementation_sha256": _sha(Path(__file__).read_bytes()),
        "learner_descriptor_sha256": record["learner_descriptor_sha256"],
        "learner_receipt_sha256": learner["receipt_sha256"], "checkpoint_sha256": learner["checkpoint_sha256"],
        "source_tree_id": record["source_tree_id"], "source_hashes": learner["source_hashes"],
        "hydration": hydrated, "world_ducklake": persisted["ducklake"], "metadata_ducklake": projection,
        "files": _inventory(output), "hydrated": True, "authority": "unverified_candidate_only",
        "completion_authority": False, "proof_authority": False, "formalization_authority": False,
        "provider_calls": 0}
    digest = _write(output / "receipt.json", receipt)
    return {**receipt, "receipt_sha256": digest}


def validate_codebase_autoencoder_index(*, repository: Path, learner: dict, expected_receipt: dict) -> dict:
    repository = Path(repository).absolute()
    if (type(expected_receipt) is not dict or expected_receipt.get("schema") != SCHEMA
            or expected_receipt.get("domain") != DOMAIN or expected_receipt.get("repository") != str(repository)):
        raise ValueError("exact code autoencoder index receipt required")
    output = _output(repository, learner, expected_receipt["output"], fresh=False)
    raw = _read(output / "receipt.json")
    if _sha(raw) != expected_receipt.get("receipt_sha256") or json.loads(raw) != {k: v for k, v in expected_receipt.items() if k != "receipt_sha256"}:
        raise ValueError("code autoencoder index receipt changed")
    if (expected_receipt.get("authority") != "unverified_candidate_only"
            or expected_receipt.get("status") != "hydrated" or expected_receipt.get("hydrated") is not True
            or expected_receipt.get("world_ducklake", {}).get("status") != "projected"
            or expected_receipt.get("metadata_ducklake", {}).get("status") != "projected"
            or expected_receipt.get("index_implementation_sha256") != _sha(Path(__file__).read_bytes())
            or any(expected_receipt.get(key) is not False for key in ("completion_authority", "proof_authority", "formalization_authority"))
            or expected_receipt.get("learner_descriptor_sha256") != _sha(_json(learner))
            or expected_receipt.get("learner_receipt_sha256") != learner["receipt_sha256"]
            or expected_receipt.get("checkpoint_sha256") != learner["checkpoint_sha256"]
            or expected_receipt.get("source_hashes") != learner["source_hashes"]):
        raise ValueError("code autoencoder index learning or authority binding differs")
    if _inventory(output) != expected_receipt.get("files"):
        raise ValueError("code autoencoder index artifact inventory changed")
    checked = validate_codebase_autoencoder(repository=repository, expected_receipt=learner)
    record = _record(repository, learner, checked)
    if (record["source_tree_id"] != expected_receipt.get("source_tree_id")
            or _hydrate(output, record) != expected_receipt.get("hydration")):
        raise ValueError("code autoencoder index current hydration differs")
    if _inventory(output) != expected_receipt["files"]:
        raise ValueError("code autoencoder catalog changed during hydration")
    validate_codebase_autoencoder(repository=repository, expected_receipt=learner)
    return {"status": "verified", "hydrated": True, "domain": DOMAIN,
        "receipt_sha256": expected_receipt["receipt_sha256"], "catalog_count": expected_receipt["hydration"]["catalog_count"],
        "completion_authority": False, "proof_authority": False, "formalization_authority": False}
