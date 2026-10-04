"""Stage a closed, unqualified two-epoch setup without opening native owners.

This host-only byte transport avoids repeating already returned setup fits.
It is not native owner validation, a clean-Git attestation, or qualification.
The fresh receiving harness must reopen both owners at their original container
paths and replay source, registry and complete model lineage before scanning.
Observations are sequential; no atomic filesystem snapshot is claimed.
"""
from __future__ import annotations

import base64
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat

SCHEMA = "inventory-resume-closed-setup-seed@1"
MATERIALIZED_SCHEMA = "inventory-resume-materialized-setup-seed@1"
MAX_FILES = 4096
MAX_BYTES = 128 * 1024 * 1024
MAX_JSON_BYTES = 16 * 1024 * 1024
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_FALSE = {name: False for name in ("native_owners_opened", "training_executed", "source_execution_attested",
                                 "proof_authority", "qualification_authority")}
_PRODUCERS = {
    "ipfs_datasets_py.logic.software_contracts.codebase_source_training": "aea5fa8803ae6b6dde7c4e71b81f68a7f95257b3a5d76cea693ee13e66710aa0",
    "ipfs_datasets_py.logic.software_contracts.codebase_ir_targets": "88d619efdc83f8e207b13b05f879c956655a7b5825f5647b46457ca92539c3cf",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_runtime_registry": "ca31983a5f514c7f5cf8b1b9250ef92e1beb13c0a2305381aac575931e79b4ac",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.autoencoder_projection_features": "0d93a2e56d0b46cff34f15c4792042bca33f47af00da80e21fa501ddf442de06",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.codebase_feature_worker": "91174cb51a0832e4cd691e56f94c0d6b34af1f06e6afec2c0d780d58fa3196db",
    "ipfs_datasets_py.optimizers.logic_theorem_optimizer.modal_autoencoder_cuda": "954bcdee38c8f55f0d9d57045b9f2648aac7efcdf105b3655f410dea1249f2e1",
}
_TRAIN_FIELDS = {"authority", "checkpoint_raw_cid", "contract_sha256", "feature_space_sha256", "head",
    "model_head_selected", "parent_version_id", "registry_artifact", "report_json", "schema",
    "source_model_generation", "state_sha256", "training_performed_during_load", "variant_id", "version_id"}
_TRAIN_AUTHORITY = {"admission_authority", "admitted", "behavioral_satisfaction", "completion_authority",
    "formalized", "promotion_performed", "proof_authority", "qualified", "source_runtime_semantics_verified"}
_RECEIPT_FIELDS = {"schema", "qualified", "source_namespace", "source_pins", "producer_pins", "copied_members",
    "copied_bytes", "copied_files", "head", "root_version_id", "child_version_id", "checkpoint_states",
    "inherited_actual_setup_epochs", "new_fitting_epochs", "unknown_fitting_epochs", "authority"}


class ClosedSetupError(ValueError):
    """Closed setup, bounded byte membership, or fresh-copy guards differ."""


def _require(value, message):
    if not value:
        raise ClosedSetupError(message)


def _wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_wire(value)).hexdigest()


def _integer(value, maximum, minimum=0):
    _require(type(value) is int and minimum <= value <= maximum, "bounded exact integer required")


def _closed(value, fields, name):
    _require(type(value) is dict and set(value) == set(fields), "closed " + name + " required")


def _relative(value):
    _require(type(value) is str and 0 < len(value.encode()) <= 2048, "bounded relative path required")
    path = Path(value)
    _require(not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)
             and path.as_posix() == value, "exact relative path required")
    return path


def _absolute(path, *, exists=True):
    _require(isinstance(path, Path) and ".." not in path.parts, "exact filesystem Path required")
    path = path.absolute()
    parent = path if exists else path.parent
    _require(parent.resolve(strict=True) == parent, "filesystem ancestor alias refused")
    if exists:
        _require(stat.S_ISDIR(path.lstat().st_mode), "exact directory required")
    return path


def _fingerprint(info):
    return (info.st_dev, info.st_ino, info.st_mode, info.st_nlink, info.st_size,
            info.st_mtime_ns, info.st_ctime_ns)


def _regular(path):
    _require(path.parent.resolve(strict=True) == path.parent, "file ancestor alias refused")
    info = path.lstat()
    _require(stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and not info.st_mode & 0o7000,
             "single-link ordinary regular file required")
    _integer(info.st_size, MAX_BYTES)
    return info


class _ObservationLedger(dict):
    """Body-free aggregate reservation precedes every first file read."""

    def __init__(self):
        super().__init__()
        self.reserved = {}

    def reserve(self, paths):
        for path in paths:
            identity = _fingerprint(_regular(path))
            if path in self.reserved:
                _require(self.reserved[path] == identity, "reserved source identity changed")
            else:
                self.reserved[path] = identity
            _require(len(self.reserved) <= MAX_FILES
                     and sum(info[4] for info in self.reserved.values()) <= MAX_BYTES,
                     "aggregate observed setup bound exceeded before body read")


def _open_read(path):
    """Resolve every component through non-following directory descriptors."""
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for part in path.parts[1:-1]:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = child
        return os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)


def _read(path, maximum=MAX_BYTES, *, expected=None, keep=True):
    before = _regular(path)
    _require(before.st_size <= maximum and (expected is None or _fingerprint(before) == expected),
             "file bound or initial identity differs")
    descriptor = _open_read(path)
    digest, chunks, size = hashlib.sha256(), [], 0
    try:
        _require(_fingerprint(os.fstat(descriptor)) == _fingerprint(before), "opened file identity differs")
        while block := os.read(descriptor, 1024 * 1024):
            size += len(block)
            _require(size <= maximum, "file grew beyond its byte bound")
            digest.update(block)
            if keep:
                chunks.append(block)
        _require(_fingerprint(os.fstat(descriptor)) == _fingerprint(before)
                 and _fingerprint(_regular(path)) == _fingerprint(before) and size == before.st_size,
                 "file changed during guarded read")
    finally:
        os.close(descriptor)
    return (b"".join(chunks) if keep else None), {"bytes": size, "sha256": digest.hexdigest()}, _fingerprint(before)


def _json(path):
    raw, pin, identity = _read(path, MAX_JSON_BYTES)
    def pairs(items):
        value = {}
        for key, item in items:
            _require(key not in value, "duplicate JSON key refused")
            value[key] = item
        return value
    def nonfinite(value):
        raise ClosedSetupError("finite JSON required")
    def finite_float(value):
        result = float(value)
        _require(math.isfinite(result), "finite JSON required")
        return result
    try:
        return json.loads(raw, object_pairs_hook=pairs, parse_constant=nonfinite, parse_float=finite_float), pin, identity
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ClosedSetupError("bounded finite JSON required") from error


def _tree(root):
    _require(root.resolve(strict=True) == root, "tree ancestor alias refused")
    files, directories, stack, seen = {}, {}, [root], set()
    while stack:
        directory = stack.pop()
        info = directory.lstat()
        _require(stat.S_ISDIR(info.st_mode) and len(directory.relative_to(root).parts) <= 64,
                 "bounded ordinary directory tree required")
        key = (info.st_dev, info.st_ino)
        _require(key not in seen, "directory alias refused")
        seen.add(key)
        directories[directory.relative_to(root).as_posix()] = _fingerprint(info)
        _require(len(directories) <= MAX_FILES, "directory traversal bound exceeded")
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                relative = path.relative_to(root).as_posix()
                _relative(relative)
                if stat.S_ISDIR(entry.stat(follow_symlinks=False).st_mode):
                    stack.append(path)
                    _require(len(stack) + len(directories) <= MAX_FILES, "directory traversal bound exceeded")
                else:
                    files[relative] = _fingerprint(_regular(path))
                    _require(len(files) <= MAX_FILES, "file traversal bound exceeded")
    _require(sum(value[4] for value in files.values()) <= MAX_BYTES, "tree byte bound exceeded")
    return files, directories


def _baseline(rows):
    _require(type(rows) is list and 0 < len(rows) <= MAX_FILES, "bounded baseline members required")
    result = {}
    for row in rows:
        _closed(row, {"path", "bytes", "sha256"}, "artifact baseline")
        _relative(row["path"])
        _integer(row["bytes"], MAX_BYTES)
        _require(type(row["sha256"]) is str and _SHA.fullmatch(row["sha256"])
                 and row["path"] not in result, "unique SHA-bound artifact member required")
        result[row["path"]] = row
    return result


def _head(value):
    _closed(value, {"schema", "repository_id", "generation", "manifest_cid", "snapshot_cid", "ast_revision_id", "receipt_cid"}, "source head")
    _require(value["schema"] == "codebase-head@1" and value["repository_id"] == "qualification:inventory-resume",
             "authored qualification source head required")
    _integer(value["generation"], 2**63 - 1, 1)
    for key in ("manifest_cid", "snapshot_cid", "receipt_cid"):
        _require(type(value[key]) is str and re.fullmatch(r"baguqeera[a-z2-7]{52}", value[key]), "bounded structural head CID required")
    _require(value["ast_revision_id"] == "rev:" + value["repository_id"] + ":snapshot:" + value["snapshot_cid"],
             "source revision binding differs")


def _closed_run(container, result):
    _require(type(container) is dict and container.get("schema") == "inventory-resume-worker-offline-container-execution@1",
             "closed container-final record required")
    for name in ("container_removed", "host_reservation_released", "container_results_copied",
                 "retained_source_verified_after_execution", "runtime_used_retained_source_copies", "native_module_launched"):
        _require(container.get(name) is True, "closed container lifecycle flag required: " + name)
    _require(type(container.get("container_id")) is str and re.fullmatch(r"[0-9a-f]{64}", container["container_id"])
             and type(container.get("image_id")) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", container["image_id"]),
             "exact retained container/image identities required")
    resources = container.get("host_resources_after_cleanup")
    _require(type(resources) is dict, "closed host resources required")
    for name in ("active_lease_count", "waiting_request_count", "allocated_child_process_slots"):
        _integer(resources.get(name), 0)
    for name in ("cpu_slots", "memory_mb"):
        _integer(resources.get("allocated", {}).get(name), 0)
    _require(type(result) is dict and result.get("schema") == "codebase-inventory-resume-native-qualification@1"
             and result.get("qualified") is False and result.get("unknown_fitting_epochs") is False,
             "unqualified closed setup with known fitting epochs required")
    _integer(result.get("known_actual_setup_epochs"), 2, 2)
    _integer(result.get("post_setup_fit_attempt_count"), 0)
    for name in ("proof_authority", "source_execution_attested", "scan_execution_attested", "production_default_activated"):
        _require(result.get(name) is False, "setup authority must remain false")
    _require(not any(any(marker in name.lower() for marker in ("worker", "publication", "published", "admission"))
                     for name in result if name != "scope"), "worker/publication result cannot be reused")
    attempts = result.get("setup_training_attempts")
    _require(type(attempts) is list and len(attempts) == 2, "exact two returned setup fits required")
    for attempt, name in zip(attempts, ("root", "child")):
        _closed(attempt, {"name", "requested_epochs", "actual_completed_epochs", "unknown_actual_epochs_on_failure", "version_id"}, "returned setup attempt")
        _require(attempt["name"] == name and attempt["unknown_actual_epochs_on_failure"] is False, "returned setup attempt required")
        _integer(attempt["requested_epochs"], 1, 1)
        _integer(attempt["actual_completed_epochs"], 1, 1)
        _require(type(attempt["version_id"]) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", attempt["version_id"]), "model version identity required")
    phases = result.get("phases")
    _require(type(phases) is list and len(phases) <= 128 and all(type(row) is dict and type(row.get("name")) is str for row in phases), "bounded recorded phases required")
    _require(not any(any(marker in row["name"].lower() for marker in ("worker", "publication", "published", "admission", "signed", "materializ")) for row in phases), "worker/publication phase cannot be reused")
    for name in ("fit_private_one_epoch_root", "fit_private_same_head_one_epoch_child"):
        matches = [row for row in phases if row["name"] == name]
        _require(len(matches) == 1 and matches[0].get("status") == "completed", "setup fit did not return")
    for name in ("active_lease_count", "waiting_request_count"):
        _integer(result.get("final_resources", {}).get(name), 0)


def _current_producer_path(name):
    return Path(__file__).resolve().parents[4] / "ipfs_datasets" / Path(*name.split(".")).with_suffix(".py")


def _producer_pins(namespace, generation, observations):
    _require(type(generation) is dict and generation.get("schema") == "codebase-inventory-resume-selected-producers@1"
             and generation.get("execution_attestation") is False and type(generation.get("files")) is list
             and len(generation["files"]) <= 64, "bounded retained producer generation required")
    rows = generation["files"]
    _require(all(type(row) is dict and type(row.get("name")) is str for row in rows)
             and len({row["name"] for row in rows}) == len(rows), "unique selected producer names required")
    result = {}
    for name, digest in _PRODUCERS.items():
        matches = [row for row in rows if row["name"] == name]
        _require(len(matches) == 1 and matches[0].get("sha256") == digest, "frozen checkpoint producer generation differs")
        row = matches[0]
        _relative(row.get("copy"))
        _integer(row.get("bytes"), 4 * 1024 * 1024, 1)
        paths = (namespace / "native" / row["copy"], namespace / "datasets" / Path(*name.split(".")).with_suffix(".py"),
                 _current_producer_path(name))
        observations.reserve(paths)
        for path in paths:
            _, pin, identity = _read(path, 4 * 1024 * 1024, keep=False)
            _require(pin == {"bytes": row["bytes"], "sha256": digest}, "frozen retained/current checkpoint producer changed")
            observations[path] = (identity, pin)
        result[name] = {"bytes": row["bytes"], "sha256": digest}
    return result


def _raw_cid(raw):
    return "b" + base64.b32encode(b"\x01\x55\x12\x20" + hashlib.sha256(raw).digest()).decode().lower().rstrip("=")


def _checkpoint(record, name, native, observations):
    _closed(record, _TRAIN_FIELDS, "training record")
    _closed(record["authority"], _TRAIN_AUTHORITY, "training authority")
    _require(all(flag is False for flag in record["authority"].values()) and record["schema"] == "codebase-source-feature-training@1"
             and record["model_head_selected"] is False and record["training_performed_during_load"] is False,
             "non-authoritative private training record required")
    _closed(record["registry_artifact"], {"sha256", "bytes"}, "checkpoint artifact")
    artifact = record["registry_artifact"]
    _require(type(artifact["sha256"]) is str and _SHA.fullmatch(artifact["sha256"]), "checkpoint SHA required")
    path = native / "private/model-artifacts" / artifact["sha256"][:2] / artifact["sha256"]
    observations.reserve((path,))
    saved, pin, identity = _json(path)
    observations[path] = (identity, pin)
    _require(pin == artifact, "checkpoint artifact bytes differ")
    _closed(saved, {"contract", "feature_space", "report", "state"}, "raw checkpoint")
    raw, _, _ = _read(path, MAX_JSON_BYTES)
    _require(_wire(saved) == raw and _raw_cid(raw) == record["checkpoint_raw_cid"]
             and _digest(saved["state"]) == record["state_sha256"]
             and _wire(saved["report"]).decode() == record["report_json"]
             and saved["state"].get("contract_sha256") == record["contract_sha256"]
             and saved["report"].get("contract_sha256") == record["contract_sha256"]
             and _digest(saved["feature_space"]) == record["feature_space_sha256"], "raw checkpoint record bindings differ")
    state, report, space = saved["state"], saved["report"], saved["feature_space"]
    epochs = 1 if name == "root" else 2
    _integer(state.get("completed_epochs"), epochs, epochs)
    _integer(state.get("latent_width"), 8, 8)
    _integer(report.get("attempted_epochs"), 1, 1)
    _integer(report.get("selected_total_epochs"), epochs, epochs)
    _require(report.get("backend") == "native-projection-feature-autoencoder/v1", "native recorded checkpoint profile required")
    for value in (state, report):
        _require(all(value.get(key) is False for key in ("admitted", "qualified", "formalized", "promotion_performed")), "checkpoint authority differs")
    provenance = report.get("codebase_provenance")
    _require(type(provenance) is dict and provenance.get("schema") == "codebase-source-feature-lineage@1"
             and provenance.get("head") == record["head"] and provenance.get("parent_version_id") == record["parent_version_id"], "captured model provenance differs")
    _require(provenance.get("implementation") == {"files": _PRODUCERS, "sha256": _digest(_PRODUCERS),
        "scope": "listed_local_files_only_not_execution_attestation"}, "checkpoint source producer pins differ")
    adam = state.get("adam")
    _require(type(adam) is list and 1 <= len(adam) <= 1024, "bounded recorded Adam state required")
    steps = []
    for row in adam:
        _require(type(row) is dict, "recorded Adam row required")
        _integer(row.get("step"), epochs, epochs)
        steps.append(row["step"])
    columns = space.get("columns")
    _require(type(columns) is list and 1 <= len(columns) <= 1024, "bounded global feature basis required")
    return {"artifact": artifact, "state_sha256": record["state_sha256"], "report_sha256": _digest(report),
            "completed_epochs": epochs, "adam_steps": steps, "latent_width": 8, "feature_columns": len(columns)}


def _repository(native, head, observations, tree):
    manifest_path = native / "private/source-artifacts/structured" / head["manifest_cid"][:4] / head["manifest_cid"]
    observations.reserve((manifest_path,))
    manifest, pin, identity = _json(manifest_path)
    observations[manifest_path] = (identity, pin)
    _require(type(manifest) is dict and manifest.get("schema") == "codebase-ir-structural-manifest@1"
             and manifest.get("ast_revision_id") == head["ast_revision_id"], "captured structural manifest required")
    snapshot = manifest.get("snapshot")
    _require(type(snapshot) is dict and snapshot.get("schema") == "ipfs-datasets.software-contracts.semantic-repository-snapshot@4"
             and snapshot.get("mode") == "git-clean" and snapshot.get("repository_id") == head["repository_id"]
             and snapshot.get("snapshot_cid") == head["snapshot_cid"], "captured Git source snapshot differs")
    entries = snapshot.get("entries")
    _require(type(entries) is list and len(entries) == 300, "complete original300 source population required")
    paths = set()
    for entry in entries:
        _require(type(entry) is dict and entry.get("acquisition") in {"git-object", "opaque"}
                 and entry.get("disposition") == "clean", "original captured Git member required")
        if entry["acquisition"] == "opaque":
            _require(entry.get("kind") == "opaque" and entry.get("opaque_reason") in {"undecodable", "oversized"},
                     "captured opaque member profile differs")
        relative = entry.get("path")
        _relative(relative)
        _require(relative not in paths and not relative.startswith(".git/") and relative != ".git"
                 and entry.get("raw_path_hex") == relative.encode().hex(), "exact unique captured path required")
        paths.add(relative)
        observations.reserve((native / "repository" / relative,))
        raw, pin, identity = _read(native / "repository" / relative)
        observations[native / "repository" / relative] = (identity, pin)
        _require(type(entry.get("size_bytes")) is int and entry["size_bytes"] == len(raw), "captured repository size differs")
        oid = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        _require(oid == entry.get("git_blob_oid") == entry.get("head_blob_oid"), "original Git blob source bytes differ")
        if entry.get("source_cid") is not None:
            _require(_raw_cid(raw) == entry["source_cid"], "captured raw source CID differs")
    _require({name for name in tree if not name.startswith(".git/")} == paths and any(name.startswith(".git/") for name in tree), "full original repository/Git membership differs")
    forbidden = (".git/commondir", ".git/gitdir", ".git/objects/info/alternates")
    _require(not any(name in tree for name in forbidden) and not any(name.startswith(".git/worktrees/") for name in tree), "external Git storage refused")
    raw, pin, identity = _read(native / "repository/.git/HEAD", 4096)
    observations[native / "repository/.git/HEAD"] = (identity, pin)
    _require(raw.startswith(b"ref: refs/heads/") and raw.endswith(b"\n"), "local Git branch HEAD required")
    ref = raw[5:-1].decode("ascii")
    _relative(ref)
    _require(ref.startswith("refs/heads/"), "local Git branch reference required")
    observations.reserve((native / "repository/.git" / ref,))
    raw, pin, identity = _read(native / "repository/.git" / ref, 4096)
    observations[native / "repository/.git" / ref] = (identity, pin)
    _require(raw == (snapshot["git_commit"] + "\n").encode() and re.fullmatch(r"[0-9a-f]{40}", snapshot["git_commit"]), "captured Git commit reference differs")


def _allowed(path):
    return path in {"private/source.duckdb", "private/model.duckdb", "root.json", "child.json"} or path.startswith(
        ("private/source-artifacts/", "private/model-artifacts/", "repository/"))


def _copy_members(source, destination, members, identities):
    destination = _absolute(destination, exists=destination.exists())
    _require(destination != source and destination not in source.parents and source not in destination.parents,
             "fresh disjoint destination required")
    if destination.exists():
        _require(not any(destination.iterdir()), "fresh empty destination required")
    else:
        destination.mkdir(mode=0o700)
    for row in members:
        relative = _relative(row["path"])
        origin, target = source / relative, destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _require(target.parent.resolve(strict=True) == target.parent, "destination ancestor alias refused")
        raw, pin, _ = _read(origin, expected=identities[row["path"]])
        _require(pin == {"bytes": row["bytes"], "sha256": row["sha256"]}, "copied source bytes differ")
        descriptor = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        target.chmod(row["mode"])
        _, copied, _ = _read(target, keep=False)
        _require(copied == pin, "fresh destination copy differs")
    if (destination / "private").exists():
        (destination / "private").chmod(0o700)
    for directory in (destination, *(path for path in destination.rglob("*") if path.is_dir())):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    files, _ = _tree(destination)
    _require(set(files) == {row["path"] for row in members}, "destination membership changed")
    for row in members:
        _, pin, _ = _read(destination / row["path"], expected=files[row["path"]], keep=False)
        _require(pin == {"bytes": row["bytes"], "sha256": row["sha256"]}
                 and stat.S_IMODE(files[row["path"]][2]) == row["mode"], "destination byte/mode membership differs")
    return destination


def stage_closed_setup(namespace: Path, destination: Path) -> dict:
    """Copy only an unchanged closed original setup; never open an owner."""
    namespace = _absolute(namespace)
    destination = _absolute(destination, exists=False)
    _require(not destination.exists(), "fresh staging destination required")
    _require(destination != namespace and destination not in namespace.parents and namespace not in destination.parents,
             "fresh disjoint namespace destination required")
    native, observations, source_pins = namespace / "native", _ObservationLedger(), {}
    controls = {"container_final": namespace / "container-execution-final.json", "result": native / "result.json",
        "generation": native / "generation-inputs.json", "owners": native / "owners-before-scans.json",
        "captured": native / "captured-artifacts-before-scans.json", "root": native / "root.json", "child": native / "child.json"}
    values = {}
    observations.reserve(controls.values())
    for name, path in controls.items():
        value, pin, identity = _json(path)
        observations[path] = (identity, pin)
        values[name] = value
        if name not in {"root", "child"}:
            source_pins[name] = {"path": path.relative_to(namespace).as_posix(), **pin}
    _closed_run(values["container_final"], values["result"])
    _require(not any((native / name).exists() for name in ("inventory-worker-admission.json", "inventory-worker-context.json",
        "inventory-publication.json", "completion.json", "native_runtime", "worker", "admission")), "worker/publication output cannot be reused")
    owners = values["owners"]
    _closed(owners, {"source_head", "registry", "model_artifacts"}, "pre-scan owner baseline")
    head = values["result"].get("head")
    _head(head)
    _require(owners["source_head"] == head == values["root"].get("head") == values["child"].get("head"), "setup source heads differ")
    _require(values["root"].get("parent_version_id") is None
             and values["child"].get("parent_version_id") == values["root"].get("version_id")
             and values["result"].get("selected_version_id") == values["child"].get("version_id"), "setup root/child lineage differs")
    for role, attempt in zip(("root", "child"), values["result"]["setup_training_attempts"]):
        _require(attempt["version_id"] == values[role].get("version_id"), "returned setup version differs")
    for key in ("variant_id", "contract_sha256", "feature_space_sha256"):
        _require(values["root"].get(key) == values["child"].get(key), "setup global feature basis differs")
    producer_pins = _producer_pins(namespace, values["generation"], observations)
    private = native / "private"
    private_members = {entry.name for entry in private.iterdir()}
    private_identity = _fingerprint(private.lstat())
    _require(private_members <= {"source.duckdb", "model.duckdb", "model.duckdb.owner.lock", "source-artifacts", "model-artifacts"}, "closed native private owner membership differs")
    if (private / "model.duckdb.owner.lock").exists():
        observations.reserve((private / "model.duckdb.owner.lock",))
        _, pin, identity = _read(private / "model.duckdb.owner.lock", 0)
        observations[private / "model.duckdb.owner.lock"] = (identity, pin)
    trees = {name: _tree(native / name) for name in ("private/source-artifacts", "private/model-artifacts", "repository")}
    observations.reserve(native / name / relative for name, (files, _) in trees.items() for relative in files)
    observations.reserve((private / "source.duckdb", private / "model.duckdb"))
    captured, models = _baseline(values["captured"]), _baseline(owners["model_artifacts"])
    _require(set(captured) <= set(trees["private/source-artifacts"][0])
             and set(models) == set(trees["private/model-artifacts"][0]) and len(models) == 2,
             "pre-scan source/model artifact membership differs")
    members = []
    for prefix, rows in (("private/source-artifacts", captured), ("private/model-artifacts", models)):
        for relative, expected in rows.items():
            path = native / prefix / relative
            _, pin, identity = _read(path, expected=trees[prefix][0][relative], keep=False)
            _require(pin == {"bytes": expected["bytes"], "sha256": expected["sha256"]}, "pre-scan artifact bytes changed")
            observations[path] = (identity, pin)
            members.append({"path": prefix + "/" + relative, **pin, "mode": stat.S_IMODE(identity[2])})
    for relative, identity in trees["repository"][0].items():
        _, pin, _ = _read(native / "repository" / relative, expected=identity, keep=False)
        observations[native / "repository" / relative] = (identity, pin)
        members.append({"path": "repository/" + relative, **pin, "mode": stat.S_IMODE(identity[2])})
    _repository(native, head, observations, trees["repository"][0])
    states = {name: _checkpoint(values[name], name, native, observations) for name in ("root", "child")}
    registry = owners["registry"]
    _require(type(registry) is dict and registry.get("heads") == [] and type(registry.get("versions")) is list
             and len(registry["versions"]) == 2, "unpromoted two-model registry baseline required")
    for name in ("root", "child"):
        rows = [row for row in registry["versions"] if type(row) is list and len(row) >= 5 and row[0] == values[name]["version_id"]]
        _require(len(rows) == 1 and rows[0][1] == values[name]["variant_id"] and rows[0][2] == values[name]["parent_version_id"]
                 and json.loads(rows[0][3]) == values[name]["registry_artifact"], "registry baseline checkpoint identity differs")
    for relative in ("private/source.duckdb", "private/model.duckdb", "root.json", "child.json"):
        _, pin, identity = _read(native / relative, keep=False)
        observations[native / relative] = (identity, pin)
        mode = stat.S_IMODE(identity[2]) | (0o600 if relative.endswith(".duckdb") else 0)
        members.append({"path": relative, **pin, "mode": mode})
    members.sort(key=lambda row: row["path"])
    _require(len(members) <= MAX_FILES and sum(row["bytes"] for row in members) <= MAX_BYTES
             and len(observations) <= MAX_FILES and sum(pin["bytes"] for _, pin in observations.values()) <= MAX_BYTES,
             "aggregate copied/observed setup bound exceeded")
    identities = {row["path"]: observations[native / row["path"]][0] for row in members}
    _copy_members(native, destination, members, identities)
    for name, (files, directories) in trees.items():
        _require(_tree(native / name) == (files, directories), "source tree membership/identity changed during staging")
    _require({entry.name for entry in private.iterdir()} == private_members
             and _fingerprint(private.lstat()) == private_identity, "private owner membership changed during staging")
    for path, (identity, expected) in observations.items():
        _, actual, _ = _read(path, expected=identity, keep=False)
        _require(actual == expected, "source setup changed during staging")
    receipt = {"schema": SCHEMA, "qualified": False, "source_namespace": str(namespace), "source_pins": source_pins,
        "producer_pins": producer_pins, "copied_members": members, "copied_bytes": sum(row["bytes"] for row in members),
        "copied_files": len(members), "head": head, "root_version_id": values["root"]["version_id"],
        "child_version_id": values["child"]["version_id"], "checkpoint_states": states,
        "inherited_actual_setup_epochs": 2, "new_fitting_epochs": 0, "unknown_fitting_epochs": False, "authority": dict(_FALSE)}
    _validate_receipt(receipt)
    return receipt


def _validate_receipt(receipt):
    _closed(receipt, _RECEIPT_FIELDS, "setup seed receipt")
    _closed(receipt["authority"], _FALSE, "seed authority")
    _require(receipt["schema"] == SCHEMA and receipt["qualified"] is False and receipt["unknown_fitting_epochs"] is False
             and all(value is False for value in receipt["authority"].values()), "seed remains unqualified and inert")
    _integer(receipt["inherited_actual_setup_epochs"], 2, 2)
    _integer(receipt["new_fitting_epochs"], 0)
    _head(receipt["head"])
    _require(type(receipt["source_namespace"]) is str and Path(receipt["source_namespace"]).is_absolute(), "source namespace reference required")
    _closed(receipt["source_pins"], {"container_final", "result", "generation", "owners", "captured"}, "source observation pins")
    for row in receipt["source_pins"].values():
        _closed(row, {"path", "bytes", "sha256"}, "source pin")
        _relative(row["path"])
        _integer(row["bytes"], MAX_JSON_BYTES)
        _require(type(row["sha256"]) is str and _SHA.fullmatch(row["sha256"]), "source observation SHA required")
    for key in ("root_version_id", "child_version_id"):
        _require(type(receipt[key]) is str and re.fullmatch(r"sha256:[0-9a-f]{64}", receipt[key]), "exact seed model version required")
    _require(receipt["root_version_id"] != receipt["child_version_id"], "distinct root/child versions required")
    _closed(receipt["checkpoint_states"], {"root", "child"}, "recorded checkpoint states")
    for role, value in receipt["checkpoint_states"].items():
        _closed(value, {"artifact", "state_sha256", "report_sha256", "completed_epochs", "adam_steps", "latent_width", "feature_columns"}, "checkpoint state metadata")
        _closed(value["artifact"], {"bytes", "sha256"}, "checkpoint state artifact")
        _integer(value["artifact"]["bytes"], MAX_JSON_BYTES, 1)
        for digest in (value["state_sha256"], value["report_sha256"], value["artifact"]["sha256"]):
            _require(type(digest) is str and _SHA.fullmatch(digest), "checkpoint state SHA required")
        epochs = 1 if role == "root" else 2
        _integer(value["completed_epochs"], epochs, epochs)
        _integer(value["latent_width"], 8, 8)
        _integer(value["feature_columns"], 1024, 1)
        _require(type(value["adam_steps"]) is list and 1 <= len(value["adam_steps"]) <= 1024, "bounded Adam step metadata required")
        for step in value["adam_steps"]:
            _integer(step, epochs, epochs)
    _closed(receipt["producer_pins"], _PRODUCERS, "frozen producer pins")
    for name, expected in _PRODUCERS.items():
        row = receipt["producer_pins"][name]
        _closed(row, {"bytes", "sha256"}, "producer pin")
        _integer(row["bytes"], 4 * 1024 * 1024, 1)
        _require(row["sha256"] == expected, "seed frozen producer differs")
    rows = receipt["copied_members"]
    _require(type(rows) is list and 0 < len(rows) <= MAX_FILES, "bounded exact seed membership required")
    names = []
    for row in rows:
        _closed(row, {"path", "bytes", "sha256", "mode"}, "copied seed member")
        _relative(row["path"])
        _require(_allowed(row["path"]) and type(row["sha256"]) is str and _SHA.fullmatch(row["sha256"]), "allowed SHA-bound seed member required")
        _integer(row["bytes"], MAX_BYTES)
        _integer(row["mode"], 0o777)
        if row["path"] in {"private/source.duckdb", "private/model.duckdb"}:
            _require(row["mode"] & 0o600 == 0o600, "materialized database must be owner-writable")
        names.append(row["path"])
    _require(names == sorted(set(names)) and all(name in names for name in ("root.json", "child.json", "private/source.duckdb", "private/model.duckdb")), "complete unique seed members required")
    _integer(receipt["copied_files"], MAX_FILES, 1)
    _integer(receipt["copied_bytes"], MAX_BYTES)
    _require(receipt["copied_files"] == len(rows) and receipt["copied_bytes"] == sum(row["bytes"] for row in rows), "seed copied totals differ")
    _wire(receipt)


def materialize_staged_setup(seed: Path, output: Path, receipt: dict) -> dict:
    """Copy exact seed bytes into a fresh/empty native output; fit nothing."""
    _validate_receipt(receipt)
    seed = _absolute(seed)
    files, directories = _tree(seed)
    _require(set(files) == {row["path"] for row in receipt["copied_members"]}, "staged seed membership differs")
    observations = _ObservationLedger()
    observations.reserve(seed / relative for relative in files)
    records = {}
    for role in ("root", "child"):
        records[role], pin, identity = _json(seed / (role + ".json"))
        observations[seed / (role + ".json")] = (identity, pin)
        _require(records[role].get("head") == receipt["head"] and records[role].get("version_id") == receipt[role + "_version_id"], "seed training identity differs")
        _require(_checkpoint(records[role], role, seed, observations) == receipt["checkpoint_states"][role], "seed checkpoint state metadata differs")
    _require(records["root"]["parent_version_id"] is None and records["child"]["parent_version_id"] == receipt["root_version_id"], "seed parent lineage differs")
    repository = {name.removeprefix("repository/"): info for name, info in files.items() if name.startswith("repository/")}
    _repository(seed, receipt["head"], observations, repository)
    destination = _copy_members(seed, output, receipt["copied_members"], files)
    _require(_tree(seed) == (files, directories), "staged seed changed during materialization")
    for row in receipt["copied_members"]:
        _, pin, _ = _read(seed / row["path"], expected=files[row["path"]], keep=False)
        _require(pin == {"bytes": row["bytes"], "sha256": row["sha256"]}, "staged seed bytes changed")
    return {"schema": MATERIALIZED_SCHEMA, "qualified": False, "seed_receipt_sha256": _digest(receipt),
        "output": str(destination), "copied_members": json.loads(_wire(receipt["copied_members"])),
        "head": receipt["head"], "root_version_id": receipt["root_version_id"], "child_version_id": receipt["child_version_id"],
        "checkpoint_states": receipt["checkpoint_states"], "inherited_actual_setup_epochs": 2,
        "new_fitting_epochs": 0, "unknown_fitting_epochs": False, "authority": dict(_FALSE)}
