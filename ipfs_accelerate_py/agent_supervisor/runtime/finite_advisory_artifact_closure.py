"""A bounded advisory byte closure for an already admitted finite worker.

Preparation replays actual native contexts and the mathematical lowering. The
live capability then checks only detached bytes at process birth. A serialized
reference cannot grant launch; signed model-off task guards retain authority.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from ipfs_datasets_py.logic.software_contracts.content import (
    canonical_dag_json_bytes, cid_for_bytes, cid_for_structured,
)
from ..control.profile_authority import verify_did_key_signature

SCHEMA = "supervisor-finite-advisory-artifact-closure@1"
PROFILE = "finite-repository-advisory-artifact-closure@1"
REFERENCE_SCHEMA = "supervisor-finite-advisory-artifact-closure-reference@1"
SCOPE = "prepared_advisory_bytes_at_existing_native_worker_process_birth"
LIMITS = {"files": 512, "directories": 128, "file_bytes": 64 * 1024**2,
          "total_bytes": 128 * 1024**2, "material_bytes": 128 * 1024,
          "inventory_entries": 1024, "path_bytes": 4096}
CLAIMS = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "proof_authority",
    "execution_authority", "publication_authority", "completion_authority",
    "omission_authority", "production_activated", "convergence_proved",
    "generalization_verified", "parser_correctness_proved",
    "universal_python_semantics_proved", "formal_decoder_available",
    "model_influenced_worker_edit", "authenticated_process_origin",
    "atomicity_attested", "model_enabled_for_signed_worker",
)}
BRIDGE_CLAIMS = {key: value for key, value in CLAIMS.items() if key not in {
    "authenticated_process_origin", "atomicity_attested", "model_enabled_for_signed_worker"}}
BRIDGE_FIELDS = {"schema", "scope", "admission_cid", "reviewed_candidate_cid",
    "generated_result_cid", "worker_candidate_cid", "head", "source_cid", "replacement_cid",
    "replacement_sha256", "task_cid", "task_revision", "administrator_task_cids",
    "model_binding", "model_context_cid", "model_version_id", "training_steps",
    "provider_calls", "bridge_cid", *BRIDGE_CLAIMS}
CANDIDATE_FIELDS = {"artifact", "sha256", "candidate_cid", "finite_admission_cid",
    "semantic_context_cid", "task_cid", "task_id", "task_revision", "before_sha256", "after_sha256"}
HEAD_FIELDS = {"schema", "repository_id", "generation", "manifest_cid", "snapshot_cid",
               "ast_revision_id", "receipt_cid"}
CONTEXT_FIELDS = {"schema", "profile", "mode", "head", "structural_context_cid",
    "semantic_state_cid", "model_enabled", "version_id", "parent_version_id", "model_record_cid",
    "candidate_checkpoint_raw_cid", "feature_space_sha256", "contract_sha256", "state_sha256",
    "latent_width", "parameter_dtype", "actual_training_delta", "selected_total_epochs",
    "selected_optimizer_steps", "selection_version_cid", "metrics_scope", "representation",
    "evaluation_scope", "receipt_scope", "artifacts", "authority", "context_cid"}
CONTEXT_ROLES = {"invocation", "record", "checkpoint", "lineage", "inference", "metrics"}
CONTEXT_AUTHORITY = {"source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
    "proof_authority", "execution_authority", "completion_authority", "mutation_authority",
    "admission_authority", "qualified", "formalized", "promotion_performed",
    "behavioral_satisfaction", "formal_decoder_available", "convergence_proved"}
SOURCE_FIELDS = {"source_cid", "git_commit", "manifest_cid", "ast_revision_id",
                 "semantic_state_cid", "custody_cid"}
MODEL_FIELDS = {"training_context", "frozen_context"}
LOWERING_FIELDS = {"result_cid", "translation_cid", "contract_cid", "tool_policy_cid",
    "process_policy_cid", "lean_certificate_cid", "lean_olean_cid", "source_cid", "status", "scope"}
PROPOSAL_FIELDS = {"reviewed_candidate_cid", "generated_result_cid", "worker_candidate_cid",
                  "replacement_cid", "replacement_sha256", "bridge"}
PAYLOAD_FIELDS = {"schema", "profile", "scope", "head", "finite_admission_cid",
    "semantic_context_cid", "candidate", "administrator_task_cids", "source", "model", "lowering",
    "proposal", "limits", "files", "directories", "absent_paths", "implementation", "authority"}
REFERENCE_FIELDS = {"schema", "profile", "closure_cid", "signed_closure", "artifact", "authority"}
FILE_FIELDS = {"path", "size_bytes", "sha256", "witness"}
DIRECTORY_FIELDS = {"path", "witness", "entries"}
_SHA = re.compile(r"[0-9a-f]{64}")
_SEAL = object()


class FiniteAdvisoryArtifactClosureError(ValueError):
    """Prepared identities or detached physical bytes differ."""


def _need(value, message):
    if not value:
        raise FiniteAdvisoryArtifactClosureError(message)


def _wire(value):
    try:
        return canonical_dag_json_bytes(value)
    except (ValueError, TypeError, RecursionError) as error:
        raise FiniteAdvisoryArtifactClosureError("bounded canonical advisory material required") from error


def _plain(value):
    pending, nodes = [(value, 0)], 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= 50000 and depth <= 32, "advisory material JSON structure exceeds bound")
        if type(item) is dict:
            _need(len(item) <= 1024 and all(type(key) is str for key in item),
                  "bounded string-key advisory mapping required")
            pending.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            _need(len(item) <= LIMITS["inventory_entries"], "advisory JSON list exceeds bound")
            pending.extend((child, depth + 1) for child in item)
        else:
            _need(type(item) in {str, int, bool, type(None)}, "advisory references cannot contain numerical floats")
            if type(item) is str:
                _need(len(item.encode()) <= LIMITS["material_bytes"], "advisory string exceeds material bound")
    raw = _wire(value)
    _need(len(raw) <= LIMITS["material_bytes"], "advisory material exceeds 128 KiB")
    return json.loads(raw)


def _same(left, right):
    return _wire(left) == _wire(right)


def _digest(value):
    _need(type(value) is str and _SHA.fullmatch(value) is not None, "exact SHA256 required")


def _token(value):
    _need(type(value) is str and 0 < len(value.encode()) <= 1024, "bounded identity token required")


def _path(value, *, existing=True):
    _need(type(value) in {str, Path} or isinstance(value, Path), "literal artifact path required")
    path = Path(value)
    _need(path.is_absolute() and len(os.fsencode(path)) <= LIMITS["path_bytes"]
          and path.resolve(strict=existing) == path and not path.is_symlink(),
          "canonical absolute non-symlink artifact path required")
    return path


def _witness(info):
    return [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns]


def _read(path, expected=None):
    path = _path(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
              and 0 <= before.st_size <= LIMITS["file_bytes"],
              "bounded regular advisory artifact required")
        if expected is not None:
            _need(_witness(before) == expected["witness"], "advisory artifact inode or metadata changed")
        digest, blocks, remaining = hashlib.sha256(), [], before.st_size
        while remaining:
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            _need(bool(block), "advisory artifact shortened")
            blocks.append(block)
            digest.update(block)
            remaining -= len(block)
        _need(not os.read(descriptor, 1), "advisory artifact grew")
        after, current = os.fstat(descriptor), path.stat(follow_symlinks=False)
        _need(_witness(before) == _witness(after) == _witness(current)
              and path.resolve(strict=True) == path and not path.is_symlink(),
              "advisory descriptor or canonical path changed during read")
        pin = {"path": str(path), "size_bytes": before.st_size,
               "sha256": digest.hexdigest(), "witness": _witness(before)}
        if expected is not None:
            _need(pin == expected, "advisory artifact bytes changed")
        return pin, b"".join(blocks)
    finally:
        os.close(descriptor)


def _directory(path):
    path = _path(path)
    before = path.stat(follow_symlinks=False)
    _need(stat.S_ISDIR(before.st_mode), "regular advisory directory required")
    names = []
    with os.scandir(path) as entries:
        for item in entries:
            _need(len(names) < LIMITS["inventory_entries"], "advisory directory inventory exceeds bound")
            info = item.stat(follow_symlinks=False)
            _need(not item.is_symlink() and (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)),
                  "advisory inventory contains a symlink or nonregular entry")
            _need(len(os.fsencode(item.name)) <= LIMITS["path_bytes"], "advisory entry name exceeds bound")
            names.append({"name": item.name, "kind": "directory" if stat.S_ISDIR(info.st_mode) else "file"})
    current = path.stat(follow_symlinks=False)
    witness = lambda info: [info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid]
    _need(witness(before) == witness(current) and path.resolve(strict=True) == path,
          "advisory directory inode changed during inventory")
    return {"path": str(path), "witness": witness(before), "entries": sorted(names, key=lambda row: row["name"])}


def _head(value):
    _need(type(value) is dict and set(value) == HEAD_FIELDS and value["schema"] == "codebase-head@1"
          and type(value["generation"]) is int and 1 <= value["generation"] <= 2**63 - 1,
          "closed exact native source head required")
    for key in HEAD_FIELDS - {"schema", "generation"}:
        _token(value[key])


def _context_pair(model, head):
    _need(type(model) is dict and set(model) == MODEL_FIELDS, "closed train/frozen model pair required")
    for key, mode, delta in (("training_context", "train", 16), ("frozen_context", "frozen", 0)):
        row = model[key]
        _need(type(row) is dict and set(row) == CONTEXT_FIELDS
              and row["schema"] == "supervisor-codebase-feature-context@1"
              and row["profile"] == "codebase_ir/source_bound_feature_v1"
              and row["mode"] == mode and row["model_enabled"] is True
              and type(row["actual_training_delta"]) is int and row["actual_training_delta"] == delta
              and type(row["latent_width"]) is int and row["latent_width"] == 8
              and row["parameter_dtype"] == "float64" and _same(row["head"], head)
              and type(row["authority"]) is dict and set(row["authority"]) == CONTEXT_AUTHORITY
              and all(value is False for value in row["authority"].values())
              and row["context_cid"] == cid_for_structured({k: v for k, v in row.items() if k != "context_cid"}),
              "exact nonauthoritative sixteen-epoch train/frozen root contexts required")
        _need(type(row["artifacts"]) is dict and set(row["artifacts"]) == CONTEXT_ROLES,
              "closed six-role feature artifact population required")
        for role, blob in row["artifacts"].items():
            _need(type(blob) is dict and set(blob) == {"schema", "role", "relative_path", "sha256", "size_bytes", "blob_cid"}
                  and blob["schema"] == "supervisor-codebase-feature-blob@1" and blob["role"] == role
                  and blob["relative_path"] == role + ".json" and type(blob["size_bytes"]) is int
                  and 0 <= blob["size_bytes"] <= 32 * 1024**2
                  and blob["blob_cid"] == cid_for_structured({k: v for k, v in blob.items() if k != "blob_cid"}),
                  "closed role-specific feature blob required")
            _digest(blob["sha256"])
        for field in ("feature_space_sha256", "contract_sha256", "state_sha256"):
            _digest(row[field])
        _need(type(row["selected_total_epochs"]) is int and 1 <= row["selected_total_epochs"] <= 16
              and type(row["selected_optimizer_steps"]) is list and len(row["selected_optimizer_steps"]) == 4
              and all(type(step) is int and step == row["selected_total_epochs"] for step in row["selected_optimizer_steps"]),
              "root selected epochs or exact Adam steps differ")
        for field in ("version_id", "model_record_cid", "candidate_checkpoint_raw_cid", "selection_version_cid",
                      "structural_context_cid", "semantic_state_cid"):
            _token(row[field])
        _need(row["parent_version_id"] is None, "first closure requires a root model")
    train, frozen = model["training_context"], model["frozen_context"]
    shared = CONTEXT_FIELDS - {"context_cid", "mode", "actual_training_delta", "artifacts"}
    _need(all(_same(train[k], frozen[k]) for k in shared)
          and all(train["artifacts"][k] == frozen["artifacts"][k] for k in {"checkpoint", "record", "lineage", "metrics"})
          and train["context_cid"] != frozen["context_cid"],
          "training and frozen contexts selected different source/model identities")


def _validate_payload(value):
    _need(type(value) is dict and set(value) == PAYLOAD_FIELDS and value["schema"] == SCHEMA
          and value["profile"] == PROFILE and value["scope"] == SCOPE and value["limits"] == LIMITS
          and type(value["authority"]) is dict and set(value["authority"]) == set(CLAIMS)
          and all(flag is False for flag in value["authority"].values()), "closed advisory closure payload required")
    _head(value["head"])
    for name in ("finite_admission_cid", "semantic_context_cid"):
        _token(value[name])
    candidate, population = value["candidate"], value["administrator_task_cids"]
    _need(type(candidate) is dict and set(candidate) == CANDIDATE_FIELDS
          and type(candidate["task_revision"]) is int and 1 <= candidate["task_revision"] <= 2**63 - 1
          and candidate["finite_admission_cid"] == value["finite_admission_cid"]
          and candidate["semantic_context_cid"] == value["semantic_context_cid"]
          and type(population) is list and len(population) == 2 and population == sorted(set(population))
          and candidate["task_cid"] in population, "exact original two-task finite candidate required")
    for key in ("sha256", "before_sha256", "after_sha256"):
        _digest(candidate[key])
    for key in CANDIDATE_FIELDS - {"task_revision", "sha256", "before_sha256", "after_sha256"}:
        _token(candidate[key])
    for item in population:
        _token(item)
    source = value["source"]
    _need(type(source) is dict and set(source) == SOURCE_FIELDS
          and source["manifest_cid"] == value["head"]["manifest_cid"]
          and source["ast_revision_id"] == value["head"]["ast_revision_id"]
          and type(source["git_commit"]) is str and re.fullmatch(r"[0-9a-f]{40}", source["git_commit"]),
          "closed source/Git identity required")
    for key in SOURCE_FIELDS - {"git_commit"}:
        _token(source[key])
    _context_pair(value["model"], value["head"])
    _need(source["semantic_state_cid"] == value["model"]["frozen_context"]["semantic_state_cid"],
          "model semantic state differs from admitted source")
    lowering = value["lowering"]
    _need(type(lowering) is dict and set(lowering) == LOWERING_FIELDS
          and lowering["scope"] == "formal_ast_lowering_correctness"
          and lowering["status"] == "model_refuted" and lowering["source_cid"] == source["source_cid"],
          "closed initial refuted mathematical AST lowering required")
    for key in LOWERING_FIELDS - {"status", "scope"}:
        _token(lowering[key])
    proposal = value["proposal"]
    _need(type(proposal) is dict and set(proposal) == PROPOSAL_FIELDS
          and proposal["worker_candidate_cid"] == candidate["candidate_cid"]
          and proposal["replacement_sha256"] == candidate["after_sha256"], "closed proposal/worker identities required")
    _digest(proposal["replacement_sha256"])
    for key in PROPOSAL_FIELDS - {"bridge", "replacement_sha256"}:
        _token(proposal[key])
    bridge = proposal["bridge"]
    _need(type(bridge) is dict and set(bridge) == BRIDGE_FIELDS
          and bridge["schema"] == "finite-repository-advisory-worker-bridge@1"
          and bridge["scope"] == "advisory_generated_bytes_to_separately_admitted_native_worker_candidate"
          and all(bridge[key] is False for key in BRIDGE_CLAIMS)
          and bridge["bridge_cid"] == cid_for_structured({k: v for k, v in bridge.items() if k != "bridge_cid"})
          and type(bridge["task_revision"]) is int and bridge["task_revision"] == candidate["task_revision"]
          and all(type(bridge[key]) is int and bridge[key] == 0 for key in ("training_steps", "provider_calls")),
          "closed nonauthoritative original advisory bridge required")
    expected = {"admission_cid": value["finite_admission_cid"], "head": value["head"],
        "source_cid": source["source_cid"], "task_cid": candidate["task_cid"],
        "administrator_task_cids": population, "model_binding": value["model"]["training_context"],
        "model_context_cid": value["model"]["training_context"]["context_cid"],
        "model_version_id": value["model"]["training_context"]["version_id"],
        **{key: proposal[key] for key in PROPOSAL_FIELDS - {"bridge"}}}
    _need(all(_same(bridge[key], wanted) for key, wanted in expected.items()),
          "bridge retargeted a model, source, task, reviewed proposal or worker replacement")
    files, directories, absent = value["files"], value["directories"], value["absent_paths"]
    _need(type(files) is list and all(type(row) is dict and set(row) == FILE_FIELDS for row in files)
          and type(directories) is list and all(type(row) is dict and set(row) == DIRECTORY_FIELDS for row in directories),
          "closed advisory file and directory rows required")
    _need(type(files) is list and 1 <= len(files) <= LIMITS["files"]
          and type(directories) is list and 1 <= len(directories) <= LIMITS["directories"]
          and type(absent) is list and len(absent) <= 8
          and [row["path"] for row in files] == sorted({row["path"] for row in files})
          and [row["path"] for row in directories] == sorted({row["path"] for row in directories})
          and absent == sorted(set(absent)), "closed unique bounded artifact inventories required")
    total, entries = 0, 0
    for row in files:
        _need(type(row) is dict and set(row) == FILE_FIELDS and type(row["size_bytes"]) is int
              and 0 <= row["size_bytes"] <= LIMITS["file_bytes"]
              and type(row["witness"]) is list and len(row["witness"]) == 9
              and all(type(item) is int and 0 <= item <= 2**64 - 1 for item in row["witness"])
              and stat.S_ISREG(row["witness"][2]) and row["witness"][5] == 1
              and row["witness"][6] == row["size_bytes"], "closed advisory file witness required")
        _token(row["path"])
        _digest(row["sha256"])
        total += row["size_bytes"]
    _need(total <= LIMITS["total_bytes"], "advisory artifact population exceeds 128 MiB")
    for row in directories:
        _need(type(row) is dict and set(row) == DIRECTORY_FIELDS and type(row["witness"]) is list
              and len(row["witness"]) == 5 and all(type(item) is int and item >= 0 for item in row["witness"])
              and stat.S_ISDIR(row["witness"][2]) and type(row["entries"]) is list,
              "closed advisory directory witness required")
        _token(row["path"])
        names = []
        for entry in row["entries"]:
            _need(type(entry) is dict and set(entry) == {"name", "kind"}
                  and type(entry["name"]) is str and entry["name"] not in {"", ".", ".."}
                  and "/" not in entry["name"] and "\x00" not in entry["name"]
                  and entry["kind"] in {"file", "directory"}, "closed advisory directory entry required")
            names.append(entry["name"])
        _need(names == sorted(set(names)), "duplicate or unordered advisory directory entry")
        entries += len(names)
    _need(entries <= LIMITS["inventory_entries"], "combined advisory directory entries exceed bound")
    for path in absent:
        _token(path)
        _need(path not in {row["path"] for row in files}, "absent path is also a required file")
    implementation = value["implementation"]
    _need(type(implementation) is dict and 1 <= len(implementation) <= 128,
          "bounded selected producer identities required")
    for name, digest in implementation.items():
        _token(name)
        _digest(digest)
    return value


def verify_finite_advisory_artifact_closure_reference(reference, *, expected_binding=None):
    """Pure signed historical integrity; this never creates a live capability."""
    reference = _plain(reference)
    _need(type(reference) is dict and set(reference) == REFERENCE_FIELDS
          and reference["schema"] == REFERENCE_SCHEMA and reference["profile"] == PROFILE
          and type(reference["authority"]) is dict and reference["authority"] == CLAIMS
          and all(value is False for value in reference["authority"].values()),
          "closed inert advisory reference required")
    envelope = reference["signed_closure"]
    _need(type(envelope) is dict and set(envelope) == {"payload", "binding"}
          and len(_wire(envelope)) <= LIMITS["material_bytes"]
          and reference["closure_cid"] == cid_for_structured(envelope), "signed advisory closure identity differs")
    binding = envelope["binding"]
    _need(type(binding) is dict and set(binding) == {"identity", "profile_id", "signature"},
          "closed advisory signer binding required")
    for item in binding.values():
        _token(item)
    if expected_binding is not None:
        _need(type(expected_binding) is dict and set(expected_binding) in (
            {"identity", "profile_id"}, {"identity", "profile_id", "signature"})
            and all(binding[key] == expected_binding[key] for key in ("identity", "profile_id")),
            "advisory closure belongs to another prepared public signer")
    _validate_payload(envelope["payload"])
    verify_did_key_signature(identity_did=binding["identity"], payload=envelope["payload"], signature=binding["signature"])
    pin = reference["artifact"]
    _need(type(pin) is dict and set(pin) == FILE_FIELDS and type(pin["size_bytes"]) is int
          and pin["size_bytes"] == len(_wire(envelope)) <= LIMITS["material_bytes"]
          and pin["sha256"] == hashlib.sha256(_wire(envelope)).hexdigest()
          and type(pin["witness"]) is list and len(pin["witness"]) == 9
          and all(type(item) is int and 0 <= item <= 2**64 - 1 for item in pin["witness"])
          and stat.S_ISREG(pin["witness"][2]) and pin["witness"][5] == 1
          and pin["witness"][6] == pin["size_bytes"],
          "closed signed closure-file descriptor required")
    _token(pin["path"])
    return envelope["payload"]


@dataclass(frozen=True, slots=True, init=False)
class FrozenFiniteAdvisoryArtifactClosure:
    """Exact private capability; serialized references and subclasses are inert."""

    _seal: object
    _reference_bytes: bytes
    _signed_bytes: bytes
    _output_directory_bytes: bytes

    def __init__(self, seal, *, reference_bytes, signed_bytes, output_directory_bytes):
        _need(seal is _SEAL and type(reference_bytes) is bytes and type(signed_bytes) is bytes
              and type(output_directory_bytes) is bytes,
              "advisory closure must be prepared by its native factory")
        object.__setattr__(self, "_seal", seal)
        object.__setattr__(self, "_reference_bytes", reference_bytes)
        object.__setattr__(self, "_signed_bytes", signed_bytes)
        object.__setattr__(self, "_output_directory_bytes", output_directory_bytes)

    @property
    def material_binding(self):
        return json.loads(self._reference_bytes)

    def to_dict(self):
        return self.material_binding

    def require_detached(self, *, head, finite_admission_cid, semantic_context_cid,
                         candidate, administrator_task_cids):
        _need(type(self) is FrozenFiniteAdvisoryArtifactClosure and self._seal is _SEAL,
              "exact live prepared advisory closure required")
        reference = self.material_binding
        _need(_wire(reference) == self._reference_bytes
              and _wire(reference["signed_closure"]) == self._signed_bytes,
              "immutable advisory material changed")
        value = verify_finite_advisory_artifact_closure_reference(reference)
        expected = {"head": head, "finite_admission_cid": finite_admission_cid,
            "semantic_context_cid": semantic_context_cid, "candidate": candidate,
            "administrator_task_cids": administrator_task_cids}
        _head(head)
        _need(all(_same(value[key], item) for key, item in expected.items()),
              "advisory closure belongs to another native source, admission, population or candidate")
        output_directory = json.loads(self._output_directory_bytes)
        _need(_wire(_directory(Path(reference["artifact"]["path"]).parent)) == self._output_directory_bytes,
              "advisory closure output directory inode or complete inventory changed")
        _read(reference["artifact"]["path"], reference["artifact"])
        for row in value["directories"]:
            _need(_directory(row["path"]) == row, "advisory directory inode or complete inventory changed")
        for path in value["absent_paths"]:
            _need(not os.path.lexists(_path(path, existing=False)), "previously absent registry companion appeared")
        for row in value["files"]:
            _read(row["path"], row)
        for row in value["directories"]:
            _need(_directory(row["path"]) == row, "advisory inventory changed during detached verification")
        _read(reference["artifact"]["path"], reference["artifact"])
        _need(_directory(output_directory["path"]) == output_directory,
              "advisory closure output inventory changed during detached verification")
        return reference


def _bridge_expected(*, admission, semantic, reviewed, generated, worker, training, replacement):
    _need(_same(generated["parent_admission"], admission) and _same(generated["reviewed_candidate"], reviewed)
          and _same(worker["finite_admission"], admission)
          and _same(generated["head"], semantic["head"])
          and reviewed["payload"]["task_cid"] == generated["task_cid"] == worker["task_cid"]
          and _same(reviewed["payload"]["administrator_task_cids"], semantic["administrator_task_cids"])
          and cid_for_bytes(replacement) == generated["replacement_cid"] == reviewed["payload"]["after_cid"]
          and hashlib.sha256(replacement).hexdigest() == worker["edit"]["after_sha256"] == reviewed["payload"]["after_sha256"]
          and worker["semantic_context_cid"] == cid_for_structured(semantic),
          "advisory proposal does not join the exact native worker handoff")
    result = {"schema": "finite-repository-advisory-worker-bridge@1",
        "scope": "advisory_generated_bytes_to_separately_admitted_native_worker_candidate",
        "admission_cid": cid_for_structured(admission), "reviewed_candidate_cid": cid_for_structured(reviewed),
        "generated_result_cid": generated["result_cid"], "worker_candidate_cid": worker["candidate_cid"],
        "head": semantic["head"], "source_cid": semantic["source_cid"], "replacement_cid": cid_for_bytes(replacement),
        "replacement_sha256": hashlib.sha256(replacement).hexdigest(), "task_cid": worker["task_cid"],
        "task_revision": worker["task_revision"], "administrator_task_cids": semantic["administrator_task_cids"],
        "model_binding": training, "model_context_cid": training["context_cid"], "model_version_id": training["version_id"],
        "training_steps": 0, "provider_calls": 0, **BRIDGE_CLAIMS}
    result["bridge_cid"] = cid_for_structured(result)
    return result


def _validate_preparation(*, owner, registry, admission, candidate, training_context, frozen_context,
        lowering_proof, contract, tool_policy, reviewed_candidate, generated_candidate, bridge):
    from ..planning.repository_plan_preview import RepositoryPlanPreviewOwner
    from ..planning.codebase_feature_context import FrozenCodebaseFeatureContext, verify_current_context
    from ..planning.finite_integer_source_custody import capture_source_custody
    from ipfs_datasets_py.logic.software_contracts.codebase_integer_lowering_lean import validate_current_integer_offset_lowering
    from . import finite_repository_admission as finite
    from . import finite_repository_candidate as proposal
    from .finite_repository_candidate_runner import load_finite_repository_candidate
    _need(type(owner) is RepositoryPlanPreviewOwner, "exact native preview owner required")
    owner.__post_init__()
    _need(type(training_context) is FrozenCodebaseFeatureContext and type(frozen_context) is FrozenCodebaseFeatureContext,
          "actual immutable train and frozen feature contexts required")
    checked = finite.verify_finite_repository_admission(admission=admission)
    admission, semantic = checked["admission"], checked["semantic_context"]
    _need(admission["local_admission"] is not None and _same(owner.expected_head.to_dict(), semantic["head"])
          and admission["evidence"]["feature_context"]["mode"] == "model_off",
          "current original model-off finite worker admission required")
    owner.index.observe_current(owner.repository, expected_head=owner.expected_head, scheduler=owner.scheduler,
        parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
        timeout_seconds=owner.timeout_seconds, memory_mb=owner.memory_mb)
    custody = capture_source_custody(owner)
    train = verify_current_context(owner, registry, training_context)
    frozen = verify_current_context(owner, registry, frozen_context)
    _context_pair({"training_context": train, "frozen_context": frozen}, semantic["head"])
    validated = validate_current_integer_offset_lowering(lowering_proof, owner.index, owner.repository,
        expected_head=owner.expected_head, contract=contract, tool_policy=tool_policy,
        scheduler=owner.scheduler, parent_lease=owner.parent_lease, cancel_event=owner.cancel_event,
        timeout_seconds=owner.timeout_seconds, memory_mb=owner.memory_mb)
    lower = validated.to_dict()
    _need(lower["status"] == "model_refuted" and lower["scope"] == "formal_ast_lowering_correctness"
          and lower["source_cid"] == semantic["source_cid"], "current initial mathematical lowering required")
    reviewed = proposal.verify_finite_repository_candidate(admission=admission, candidate=reviewed_candidate)
    proposal.verify_generated_finite_repository_candidate(record=generated_candidate)
    _need(type(candidate) is dict and set(candidate) == CANDIDATE_FIELDS, "exact public handoff descriptor required")
    worker = load_finite_repository_candidate(artifact=Path(candidate["artifact"]), expected_sha256=candidate["sha256"])
    _need(candidate["candidate_cid"] == worker["candidate_cid"] and candidate["task_cid"] == worker["task_cid"]
          and candidate["task_revision"] == worker["task_revision"]
          and candidate["task_id"] == worker["task_id"]
          and candidate["before_sha256"] == worker["edit"]["before_sha256"]
          and candidate["after_sha256"] == worker["edit"]["after_sha256"], "public candidate descriptor differs from complete handoff")
    replacement = proposal._read(Path(generated_candidate["artifacts"]["replacement"]["path"]), 65536)
    _need(replacement == base64.b64decode(worker["edit"]["after_bytes_base64"], validate=True),
          "generated replacement differs from public worker edit")
    expected_bridge = _bridge_expected(admission=admission, semantic=semantic, reviewed=reviewed_candidate,
        generated=generated_candidate, worker=worker, training=train, replacement=replacement)
    _need(type(bridge) is dict and set(bridge) == BRIDGE_FIELDS and _same(bridge, expected_bridge),
          "complete original advisory bridge differs")
    custody.require_current()
    manifest = owner.index.load(owner.expected_head.manifest_cid)
    payload = {"head": semantic["head"], "finite_admission_cid": cid_for_structured(admission),
        "semantic_context_cid": cid_for_structured(semantic), "candidate": candidate,
        "administrator_task_cids": semantic["administrator_task_cids"],
        "source": {"source_cid": semantic["source_cid"], "git_commit": manifest.snapshot.git_commit,
            "manifest_cid": manifest.cid, "ast_revision_id": manifest.ast_revision_id,
            "semantic_state_cid": manifest.semantic_state.state_cid, "custody_cid": custody.material_binding["custody_cid"]},
        "model": {"training_context": train, "frozen_context": frozen},
        "lowering": {key: lower[key] for key in LOWERING_FIELDS - {"lean_certificate_cid", "lean_olean_cid"}},
        "proposal": {key: expected_bridge[key] for key in PROPOSAL_FIELDS - {"bridge"}}}
    payload["lowering"].update(lean_certificate_cid=cid_for_structured(lower["lean_certificate"]),
                               lean_olean_cid=lower["artifacts"]["lean_olean"]["cid"])
    payload["proposal"]["bridge"] = expected_bridge
    return {"payload": payload, "manifest": admission["declaration"]["payload"]["manifest"]["payload"],
            "custody": custody}


def _mandatory_paths(*, owner, registry, admission, candidate, training_context, frozen_context,
        lowering_proof, contract, tool_policy, reviewed_candidate, generated_candidate, bridge, prepared):
    from . import finite_repository_admission as finite
    from . import finite_repository_candidate as proposal
    from . import finite_repository_candidate_runner as worker
    from ..planning import codebase_feature_context as features
    from ..planning import finite_integer_source_custody as custody_module
    from . import local_planning_admission as local
    from ipfs_datasets_py.logic.software_contracts import codebase_integer_lowering_lean as lowering
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    paths, trees, absent, implementations, expected = set(), set(), set(), {}, {}
    def add(path, size=None, digest=None, raw=None):
        path = _path(path)
        paths.add(path)
        if raw is not None:
            size, digest = len(raw), hashlib.sha256(raw).hexdigest()
        if size is not None:
            _need(type(size) is int and 0 <= size <= LIMITS["file_bytes"], "mandatory artifact byte bound differs")
            _digest(digest)
            wanted = {"bytes": size, "sha256": digest}
            _need(str(path) not in expected or expected[str(path)] == wanted,
                  "conflicting mandatory artifact identities")
            expected[str(path)] = wanted
    for context in (training_context, frozen_context):
        output = context.output
        trees.add(output)
        add(output / "context.json", raw=context._material_bytes)
        for row in context.material_binding["artifacts"].values():
            add(output / row["relative_path"], row["size_bytes"], row["sha256"])
        retained = context.retained_artifacts
        add(owner.index.artifacts.path_for(context.material_binding["model_record_cid"]), raw=_wire(retained["record"]))
        # Native training publishes its record to structured CAS, while exact
        # raw checkpoints remain in the registry SHA store. Their raw CID is an
        # identity, not a claim that a second source-CAS copy was published.
        for ancestor in retained["lineage"]:
            row = ancestor["version"]["artifact"]
            add(registry.artifact_path(row), row["bytes"], row["sha256"])
        producer = retained["checkpoint"]["report"]["codebase_provenance"]["implementation"]
        _need(type(producer) is dict and type(producer.get("files")) is dict,
              "native training implementation file population required")
        implementations.update(producer["files"])
    lower = lowering_proof.to_dict()
    trees.add(Path(lower["output"]))
    add(Path(lower["output"]) / "result.json", raw=lowering_proof._wire)
    add(owner.index.artifacts.path_for(lowering_proof.cid), raw=lowering_proof._wire)
    add(owner.index.artifacts.path_for(cid_for_bytes(lowering_proof._wire), source=True), raw=lowering_proof._wire)
    for row in lower["artifacts"].values():
        add(Path(row["path"]), row["size_bytes"], row["sha256"])
        add(owner.index.artifacts.path_for(row["cid"], source=True), row["size_bytes"], row["sha256"])
    for name in ("python", "lean"):
        row = lower["tool_policy"][name]
        add(Path(row["path"]), row["size_bytes"], row["sha256"])
    frontend = lower["translation"]["frontend"]
    implementations.update(frontend.get("source_sha256", {}))
    trees.add(Path(generated_candidate["output"]))
    add(Path(generated_candidate["output"]) / "result.json", raw=_wire(generated_candidate))
    for row in generated_candidate["artifacts"].values():
        add(Path(row["path"]), row["bytes"], row["sha256"])
    # Full public handoff bytes were loaded and verified in preparation. Its
    # descriptor SHA remains the independently declared expected identity.
    pin, _ = _read(Path(candidate["artifact"]))
    add(Path(candidate["artifact"]), pin["size_bytes"], candidate["sha256"])
    for row in prepared["custody"]._files:
        if row.role.startswith(("working:", "cas:")):
            add(row.path, row.size, row.sha)
    trees.add(registry.artifact_root)
    paths.add(registry.database_path)
    for suffix in (".wal",):
        path = Path(str(registry.database_path) + suffix)
        (paths if os.path.lexists(path) else absent).add(path)
    modules = (sys.modules[__name__], finite, proposal, worker, features, custody_module, local, lowering, training)
    for module in modules:
        path = _path(Path(module.__file__))
        implementations[module.__name__] = hashlib.sha256(_read(path)[1]).hexdigest()
    for name, digest in implementations.items():
        _digest(digest)
        module = sys.modules.get(name)
        if module is not None and type(getattr(module, "__file__", None)) is str:
            path = _path(Path(module.__file__))
        else:
            # The selected numerical worker runs only in a subprocess. Resolve
            # its already validated package file without importing that worker.
            _need(re.fullmatch(r"ipfs_datasets_py(?:\.[a-zA-Z_][a-zA-Z_0-9]*)+", name) is not None,
                  "unsupported unloaded selected producer path")
            package = sys.modules.get("ipfs_datasets_py")
            _need(package is not None and type(getattr(package, "__file__", None)) is str,
                  "selected datasets package directory is unavailable")
            path = _path(Path(package.__file__).parent.joinpath(*name.split(".")[1:]).with_suffix(".py"))
        _need(hashlib.sha256(_read(path)[1]).hexdigest() == digest, "selected producer bytes changed")
        pin, _ = _read(path)
        add(path, pin["size_bytes"], digest)
    return {"paths": sorted(paths), "trees": sorted(trees), "absent_paths": sorted(absent),
            "implementation": implementations, "expected": expected}


def prepare_finite_advisory_artifact_closure(*, owner, registry, admission, candidate,
        training_context, frozen_context, lowering_proof, contract, tool_policy,
        reviewed_candidate, generated_candidate, bridge, artifact_paths, output):
    """Prepare actual native advisory evidence once, then freeze bounded bytes."""
    from . import local_planning_admission as local
    _need(type(artifact_paths) in {list, tuple} and len(artifact_paths) <= LIMITS["files"],
          "bounded explicit extra artifact paths required")
    output = _path(output, existing=False)
    _need(not output.exists() and output.parent.is_dir() and not output.is_relative_to(owner.repository),
          "fresh external advisory closure output required")
    arguments = dict(owner=owner, registry=registry, admission=admission, candidate=candidate,
        training_context=training_context, frozen_context=frozen_context, lowering_proof=lowering_proof,
        contract=contract, tool_policy=tool_policy, reviewed_candidate=reviewed_candidate,
        generated_candidate=generated_candidate, bridge=bridge)
    prepared = _validate_preparation(**arguments)
    selected = _mandatory_paths(**arguments, prepared=prepared)
    expected_mandatory = selected.get("expected", {})
    _need(type(expected_mandatory) is dict and len(expected_mandatory) <= LIMITS["files"],
          "bounded mandatory expected artifact identities required")
    for name, expected in expected_mandatory.items():
        _path(name)
        _need(type(expected) is dict and set(expected) == {"bytes", "sha256"}
              and type(expected["bytes"]) is int and 0 <= expected["bytes"] <= LIMITS["file_bytes"],
              "closed mandatory expected artifact identity required")
        _digest(expected["sha256"])
    paths = {_path(path) for path in selected["paths"]}
    # The closure implementation is mandatory even when a test explicitly
    # substitutes the native preparation helpers.
    paths.add(_path(Path(__file__)))
    _need(not any(output.is_relative_to(_path(tree)) for tree in selected["trees"]),
          "closure output must be outside retained immutable artifact trees")
    expected_extra = {}
    for value in artifact_paths:
        if type(value) is dict:
            _need(set(value) == {"path", "bytes", "sha256"} and type(value["bytes"]) is int
                  and 0 <= value["bytes"] <= LIMITS["file_bytes"], "closed extra historical artifact pin required")
            _digest(value["sha256"])
            path = _path(value["path"])
            wanted = (value["bytes"], value["sha256"])
            _need(path not in expected_extra or expected_extra[path] == wanted, "conflicting extra artifact identities")
            expected_extra[path] = wanted
        else:
            path = _path(value)
        paths.add(path)
    directories, pending, scanned = {}, list(selected["trees"]), 0
    while pending:
        path = _path(pending.pop())
        if str(path) in directories:
            continue
        _need(len(directories) < LIMITS["directories"], "advisory directory count exceeds bound")
        row = _directory(path)
        directories[str(path)] = row
        scanned += len(row["entries"])
        _need(scanned <= LIMITS["inventory_entries"], "advisory inventory scan exceeds bound")
        for entry in row["entries"]:
            child = path / entry["name"]
            if entry["kind"] == "directory":
                pending.append(child)
            else:
                paths.add(child)
        _need(len(paths) <= LIMITS["files"], "advisory file count exceeds bound")
    _need(1 <= len(paths) <= LIMITS["files"], "bounded nonempty advisory file population required")
    files, total = [], 0
    for path in sorted(paths):
        pin, _ = _read(path)
        _need(str(path) not in expected_mandatory or
              (pin["size_bytes"], pin["sha256"]) ==
              (expected_mandatory[str(path)]["bytes"], expected_mandatory[str(path)]["sha256"]),
              "mandatory artifact changed after genuine native validation")
        _need(path not in expected_extra or (pin["size_bytes"], pin["sha256"]) == expected_extra[path],
              "extra historical artifact bytes changed before sealing")
        files.append(pin)
        total += pin["size_bytes"]
        _need(total <= LIMITS["total_bytes"], "advisory file population exceeds 128 MiB")
    payload = {**prepared["payload"], "schema": SCHEMA, "profile": PROFILE, "scope": SCOPE,
        "limits": dict(LIMITS), "files": files, "directories": sorted(directories.values(), key=lambda row: row["path"]),
        "absent_paths": sorted(str(path) for path in selected.get("absent_paths", [])),
        "implementation": selected.get("implementation", {__name__: hashlib.sha256(_read(Path(__file__))[1]).hexdigest()}),
        "authority": dict(CLAIMS)}
    _validate_payload(payload)
    envelope = local._signed(payload, prepared["manifest"])
    signed_bytes = _wire(envelope)
    _need(len(signed_bytes) <= LIMITS["material_bytes"], "signed advisory closure exceeds 128 KiB")
    provisional_pin = {"path": str(output / "closure.json"), "size_bytes": len(signed_bytes),
        "sha256": hashlib.sha256(signed_bytes).hexdigest(), "witness": [2**64 - 1] * 9}
    provisional_reference = {"schema": REFERENCE_SCHEMA, "profile": PROFILE,
        "closure_cid": cid_for_structured(envelope), "signed_closure": envelope,
        "artifact": provisional_pin, "authority": dict(CLAIMS)}
    _need(len(_wire(provisional_reference)) <= LIMITS["material_bytes"],
          "complete advisory reference exceeds 128 KiB before publication")
    # The output directory is created only after all payload bounds pass.
    output.mkdir(mode=0o700)
    artifact = output / "closure.json"
    with artifact.open("xb") as stream:
        stream.write(signed_bytes)
    pin, raw = _read(artifact)
    _need(raw == signed_bytes, "retained advisory closure differs from signed bytes")
    reference = {"schema": REFERENCE_SCHEMA, "profile": PROFILE, "closure_cid": cid_for_structured(envelope),
        "signed_closure": envelope, "artifact": pin, "authority": dict(CLAIMS)}
    reference_bytes = _wire(reference)
    _need(len(reference_bytes) <= LIMITS["material_bytes"], "complete advisory reference exceeds 128 KiB")
    # Closure output itself has a complete one-file inventory, checked directly;
    # including its self-descriptor in the signed file would be circular.
    output_directory = _directory(output)
    _need(output_directory["entries"] == [{"name": "closure.json", "kind": "file"}],
          "advisory closure output inventory differs")
    if "custody" in prepared:
        prepared["custody"].require_current()
    result = FrozenFiniteAdvisoryArtifactClosure(_SEAL, reference_bytes=reference_bytes, signed_bytes=signed_bytes,
        output_directory_bytes=_wire(output_directory))
    result.require_detached(**{key: prepared["payload"][key] for key in (
        "head", "finite_admission_cid", "semantic_context_cid", "candidate", "administrator_task_cids")})
    return result


__all__ = ["SCHEMA", "PROFILE", "REFERENCE_SCHEMA", "LIMITS", "CLAIMS",
    "FiniteAdvisoryArtifactClosureError", "FrozenFiniteAdvisoryArtifactClosure",
    "prepare_finite_advisory_artifact_closure", "verify_finite_advisory_artifact_closure_reference"]
