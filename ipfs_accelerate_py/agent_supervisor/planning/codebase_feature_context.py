"""Current native structural feature context, with no decoder or fact authority.

Full numerical JSON stays in retained artifacts. Planning receives role-bound
SHA/length identities containing no floating point values. Every current use
observes source and replays the selected native model; history alone is inert.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import threading
import time
from typing import Any

from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from .repository_plan_preview import RepositoryPlanPreviewOwner
from .structural_codebase_context import structural_codebase_context

SCHEMA = "supervisor-codebase-feature-context@1"
PROFILE = "codebase_ir/source_bound_feature_v1"
MAX_ARTIFACT_BYTES = 32 * 1024 * 1024
MAX_RETAINED_BYTES = 64 * 1024 * 1024
_FALSE = {name: False for name in (
    "source_semantics_verified", "runtime_behavior_verified", "behavior_authority",
    "proof_authority", "execution_authority", "completion_authority", "mutation_authority",
    "admission_authority", "qualified", "formalized", "promotion_performed",
    "behavioral_satisfaction", "formal_decoder_available", "convergence_proved",
)}
_MODES = {"model_off", "frozen", "train"}
_TRAIN_GUARD = threading.Lock()
_ACTIVE_TRAINING = set()


class CodebaseFeatureContextError(ValueError):
    """Source, native model, retained bytes or profile controls differ."""


def _require(condition, message):
    if not condition:
        raise CodebaseFeatureContextError(message)


def _wire(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, RecursionError) as error:
        raise CodebaseFeatureContextError("bounded finite native JSON required") from error


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _safe(value):
    if type(value) is dict:
        _require(all(type(key) is str for key in value), "string material keys required")
        for child in value.values():
            _safe(child)
    elif type(value) is list:
        for child in value:
            _safe(child)
    else:
        _require(type(value) in {str, int, bool, type(None)}, "planning feature references cannot contain floats")


def _namespace(path, *, fresh):
    path = Path(path)
    _require(path.is_absolute() and path.resolve() == path and not path.is_symlink()
             and path.parent.is_dir(), "canonical absolute context output required")
    _require(not path.exists() if fresh else path.is_dir(), "context output must be new" if fresh else "retained context directory missing")
    return path


def _read(path, checkpoint=lambda: None):
    checkpoint()
    _require(path.resolve() == path and not path.is_symlink(), "retained artifact path changed")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_ARTIFACT_BYTES,
                 "bounded regular retained artifact required")
        chunks = []
        remaining = before.st_size
        while remaining:
            checkpoint()
            block = os.read(descriptor, min(remaining, 1024 * 1024))
            _require(bool(block), "retained artifact shortened")
            chunks.append(block)
            remaining -= len(block)
        after = os.fstat(descriptor)
        current = path.stat(follow_symlinks=False)
        _require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                 == (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
                 == (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns),
                 "retained artifact changed during reading")
        checkpoint()
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _artifact(role, raw):
    _require(len(raw) <= MAX_ARTIFACT_BYTES, "native artifact exceeds byte bound")
    body = {"schema": "supervisor-codebase-feature-blob@1", "role": role,
            "relative_path": role + ".json", "sha256": _sha(raw), "size_bytes": len(raw)}
    return {**body, "blob_cid": cid_for_structured(body)}


@dataclass(frozen=True, slots=True)
class FrozenCodebaseFeatureContext:
    output: Path
    _material_bytes: bytes

    def __post_init__(self):
        _namespace(self.output, fresh=False)
        _require(type(self._material_bytes) is bytes and len(self._material_bytes) <= 64 * 1024,
                 "immutable bounded feature material required")
        value = json.loads(self._material_bytes)
        _safe(value)
        _require(_wire(value) == self._material_bytes and value["schema"] == SCHEMA
                 and value["profile"] == PROFILE and value["mode"] in _MODES
                 and value["context_cid"] == cid_for_structured({key: child for key, child in value.items() if key != "context_cid"})
                 and type(value["authority"]) is dict and set(value["authority"]) == set(_FALSE)
                 and all(flag is False for flag in value["authority"].values()),
                 "context identity or authority differs")

    @property
    def material_binding(self):
        return json.loads(self._material_bytes)

    def to_dict(self):
        return self.material_binding

    @property
    def cid(self):
        return self.material_binding["context_cid"]

    @property
    def mode(self):
        return self.material_binding["mode"]

    @property
    def head(self):
        from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
        return CodebaseHead.from_dict(self.material_binding["head"])

    @property
    def version_id(self):
        return self.material_binding["version_id"]

    @property
    def actual_training_delta(self):
        return self.material_binding["actual_training_delta"]

    @property
    def retained_artifacts(self):
        return _checked_files(self)


def _checked_files(context, checkpoint=lambda: None):
    output = _namespace(context.output, fresh=False)
    binding = context.material_binding
    roles = {"invocation"} if context.mode == "model_off" else {
        "invocation", "record", "checkpoint", "lineage", "inference", "metrics"}
    _require(set(binding["artifacts"]) == roles and all(
        row["relative_path"] == role + ".json" for role, row in binding["artifacts"].items()),
        "closed role-specific retained artifact paths required")
    _require(_read(output / "context.json", checkpoint) == context._material_bytes,
             "retained context bytes differ")
    expected_names = {"context.json", *(row["relative_path"] for row in binding["artifacts"].values())}
    _require({p.name for p in output.iterdir()} == expected_names, "retained artifact inventory differs")
    values, total = {}, 0
    for role, expected in binding["artifacts"].items():
        raw = _read(output / expected["relative_path"], checkpoint)
        _require(_artifact(role, raw) == expected, "retained " + role + " artifact identity differs")
        total += len(raw)
        _require(total <= MAX_RETAINED_BYTES, "retained native payload bound exceeded")
        values[role] = json.loads(raw)
        _require(_wire(values[role]) == raw, "retained numerical JSON is not canonical")
    return values


def _owners(owner, registry):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    _require(type(owner) is RepositoryPlanPreviewOwner, "exact native preview owner required")
    owner.__post_init__()
    training._native_owners(owner.index, registry)


@contextmanager
def _current(owner, registry, deadline):
    from ipfs_datasets_py.logic.software_contracts.codebase_resources import acquire_codebase_resources
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError
    _owners(owner, registry)
    def pending(signal=None):
        if ((owner.cancel_event is not None and owner.cancel_event.is_set())
                or (owner.parent_lease is not None and (owner.parent_lease.released or owner.parent_lease.cancelled))
                or (signal is not None and signal.is_set())):
            raise LeaseCancelledError("codebase feature context cancelled")
        left = deadline - time.monotonic()
        if left <= 0:
            raise LeaseTimeoutError("codebase feature context deadline exceeded")
        return left
    pending()
    with acquire_codebase_resources(scheduler=owner.scheduler, parent_lease=owner.parent_lease,
            cancel_event=owner.cancel_event, timeout_seconds=min(30., pending()), memory_mb=owner.memory_mb) as lease:
        signal = lease.combined_cancellation_signal(owner.cancel_event)
        remaining = lambda: pending(signal)
        def controls():
            return {"scheduler": None, "parent_lease": lease, "cancel_event": signal,
                    "admission_timeout_seconds": min(30., remaining()),
                    "timeout_seconds": remaining(), "memory_mb": owner.memory_mb}
        with structural_codebase_context(owner.index, owner.repository,
                repository_id=owner.expected_head.repository_id, expected_head=owner.expected_head,
                **controls()) as structural:
            yield structural, controls, remaining, pending
            remaining()
        remaining()
    pending()


def _model(owner, registry, version_id):
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    from ipfs_datasets_py.optimizers.logic_theorem_optimizer import autoencoder_runtime_registry as runtimes
    record = training.load_codebase_feature_training(owner.index, registry, version_id).to_dict()
    lineage, current = [], version_id
    while current is not None:
        _require(len(lineage) < 8, "model ancestry exceeds profile")
        row = registry.get_version(current)
        checkpoint = runtimes._read_candidate(registry, row)
        lineage.append({"version": row, "checkpoint": checkpoint})
        current = row["parent_version_id"]
    checkpoint = lineage[0]["checkpoint"]
    _require(record["head"] == owner.expected_head.to_dict() and checkpoint["state"]["latent_width"] == 8,
             "selected model is not the current eight-dimensional source profile")
    return record, checkpoint, lineage


def _fresh_operation(registry, operation_id, parent_version_id):
    _require(type(operation_id) is str and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", operation_id),
             "fresh bounded native training operation required")
    with registry._transaction() as connection:
        old = connection.execute("SELECT COUNT(*) FROM autoencoder_control.operations WHERE operation_id=?",
            ["codebase-bootstrap:" + operation_id]).fetchone()[0]
        runs = connection.execute("SELECT COUNT(*) FROM autoencoder_control.runs WHERE run_id=?",
            ["codebase-source:" + operation_id]).fetchone()[0]
    _require(old == 0 and runs == 0, "training operation already exists; use frozen mode for replay")


def _serialize_training(function):
    """Fence duplicate local adapter invocations without holding native SQL locks."""
    @wraps(function)
    def guarded(*args, **kwargs):
        if kwargs.get("mode", "model_off") != "train":
            return function(*args, **kwargs)
        from ipfs_datasets_py.duckdb_control.autoencoder_registry import AutoencoderRegistry
        registry, operation = kwargs.get("registry"), kwargs.get("operation_id", "")
        _require(type(registry) is AutoencoderRegistry and type(operation) is str
            and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", operation), "fresh exact native training owner/token required")
        key = (registry, operation)
        with _TRAIN_GUARD:
            _require(key not in _ACTIVE_TRAINING, "training context operation already active")
            _ACTIVE_TRAINING.add(key)
        try:
            return function(*args, **kwargs)
        finally:
            with _TRAIN_GUARD:
                _ACTIVE_TRAINING.remove(key)
    return guarded


def _metrics(checkpoint):
    return {key: checkpoint["report"][key] for key in ("before", "after", "epochs",
        "train_coverage", "tuning_coverage", "attempted_epochs", "selected_total_epochs")}


def _invocation_binding(binding, retained, record, checkpoint, registry):
    invocation = retained["invocation"]
    expected = {"schema": "codebase-feature-context-invocation@1", "mode": binding["mode"],
        "head": binding["head"], "version_id": binding["version_id"] if binding["mode"] == "frozen" else None,
        "operation_id": "", "parent_version_id": None, "selections": [], "configuration": None}
    if binding["mode"] == "train":
        op = invocation.get("operation_id")
        _require(type(op) is str and re.fullmatch(r"[A-Za-z0-9._:-]{1,128}", op),
                 "native invocation operation differs")
        expected.update(operation_id=op, parent_version_id=record["parent_version_id"],
            selections=checkpoint["report"]["codebase_provenance"]["selections"],
            configuration={key: checkpoint["report"]["configuration"][key]
                           for key in ("epochs", "learning_rate", "seed")})
        if record["parent_version_id"] is None:
            with registry._transaction() as connection:
                rows = connection.execute("SELECT receipt FROM autoencoder_control.operations WHERE operation_id=? LIMIT 2",
                    ["codebase-bootstrap:" + op]).fetchall()
            _require(len(rows) == 1 and json.loads(rows[0][0])["version_id"] == record["version_id"],
                     "root invocation is not its native completed operation")
        else:
            completion = registry.get_run_completion("codebase-source:" + op)
            _require(completion is not None and completion["candidate_version"]["version_id"] == record["version_id"],
                     "child invocation is not its native completed operation")
    _require(_wire(invocation) == _wire(expected), "retained invocation differs from native model controls")
    if checkpoint is not None:
        _require(_wire(retained["metrics"]) == _wire(_metrics(checkpoint)),
                 "retained metrics differ from native selected checkpoint")


def _binding(mode, head, structural, artifacts, record=None, checkpoint=None, actual=0):
    state = None if checkpoint is None else checkpoint["state"]
    report = None if checkpoint is None else checkpoint["report"]
    result = {"schema": SCHEMA, "profile": PROFILE, "mode": mode, "head": head.to_dict(),
        "structural_context_cid": structural.cid, "semantic_state_cid": structural.semantic_state_cid,
        "model_enabled": mode != "model_off", "version_id": None if record is None else record["version_id"],
        "parent_version_id": None if record is None else record["parent_version_id"],
        "model_record_cid": None if record is None else cid_for_structured(record),
        "candidate_checkpoint_raw_cid": None if record is None else record["checkpoint_raw_cid"],
        "feature_space_sha256": None if state is None else state["feature_space_sha256"],
        "contract_sha256": None if state is None else state["contract_sha256"],
        "state_sha256": None if record is None else record["state_sha256"],
        "latent_width": 8, "parameter_dtype": "float64", "actual_training_delta": actual,
        "selected_total_epochs": 0 if state is None else state["completed_epochs"],
        "selected_optimizer_steps": [] if state is None else [item["step"] for item in state["adam"]],
        "selection_version_cid": None if report is None else cid_for_structured({"schema": "codebase-feature-selection@1",
            "head": head.to_dict(), "selections": report["codebase_provenance"]["selections"]}),
        "metrics_scope": "fixed tuning and repeated post-selection canary; structural reconstruction only",
        "representation": "native_compiler_structural_features_not_semantic_text_embeddings",
        "evaluation_scope": "source-cohort transductive diagnostics; no blind structural generalization",
        "receipt_scope": "retained selected process identities and fresh numerical replay; no authenticated process-origin or transitive environment attestation",
        "artifacts": artifacts, "authority": dict(_FALSE)}
    _safe(result)
    result["context_cid"] = cid_for_structured(result)
    return result


@_serialize_training
def prepare_codebase_feature_context(*, owner: RepositoryPlanPreviewOwner, registry,
        mode="model_off", output, version_id=None, selections=(), operation_id="",
        parent_version_id=None, epochs=3, learning_rate=.02, seed=1729):
    """Prepare one fresh retained context; fitting occurs only in explicit train mode."""
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    _require(type(owner) is RepositoryPlanPreviewOwner, "exact native preview owner required")
    deadline = time.monotonic() + owner.timeout_seconds
    _owners(owner, registry)
    _require(type(mode) is str and mode in _MODES, "explicit supported model mode required")
    _require(type(selections) in {list, tuple}, "explicit native selection sequence required")
    if mode == "train":
        _require(version_id is None and owner.memory_mb >= 1024, "training needs native source admission and no supplied candidate")
        _require(type(epochs) is int and 1 <= epochs <= 32 and type(seed) is int and 0 <= seed < 2**31
            and type(learning_rate) in {int, float} and math.isfinite(learning_rate) and 0 < learning_rate <= .1,
            "bounded native fitting configuration required")
        selections = training._selections(selections, training.CodebaseFeatureTrainingLimits())
        _fresh_operation(registry, operation_id, parent_version_id)
    else:
        _require(not selections and not operation_id and parent_version_id is None
            and type(epochs) is int and epochs == 3 and type(seed) is int and seed == 1729
            and type(learning_rate) in {int, float} and learning_rate == .02,
            "frozen/off mode cannot accept fitting controls")
        _require((version_id is None) if mode == "model_off" else (type(version_id) is str and bool(version_id) and owner.memory_mb >= 1024),
            "off mode has no model; frozen mode requires an admitted current native version")
    output = _namespace(output, fresh=True)
    excluded = (owner.repository, owner.index.artifacts.root, registry.artifact_root)
    _require(not any(output == Path(path).resolve() or Path(path).resolve() in output.parents for path in excluded),
             "context output must be outside tested source and native artifact stores")
    invocation = {"schema": "codebase-feature-context-invocation@1", "mode": mode,
        "head": owner.expected_head.to_dict(), "version_id": version_id, "operation_id": operation_id,
        "parent_version_id": parent_version_id,
        "selections": [item.to_dict() for item in selections],
        "configuration": None if mode != "train" else {"epochs": epochs, "learning_rate": float(learning_rate), "seed": seed}}
    context = None
    with _current(owner, registry, deadline) as (structural, controls, remaining, final_pending):
        output.mkdir(mode=0o700)
        invocation_raw = _wire(invocation)
        with (output / "invocation.json").open("xb") as stream:
            stream.write(invocation_raw)
        retained = {"invocation": invocation}
        record = checkpoint = None
        actual = 0
        if mode != "model_off":
            if mode == "train":
                _fresh_operation(registry, operation_id, parent_version_id)
                current = training.train_current_codebase_features(owner.index, owner.repository,
                    expected_head=owner.expected_head, registry=registry, selections=selections,
                    operation_id=operation_id, parent_version_id=parent_version_id,
                    epochs=epochs, learning_rate=learning_rate, seed=seed, **controls())
                version_id = current.to_dict()["version_id"]
            record, checkpoint, lineage = _model(owner, registry, version_id)
            if mode == "train":
                actual = checkpoint["report"]["attempted_epochs"]
            paths = [row["path"] for row in checkpoint["report"]["codebase_provenance"]["selections"]]
            inference = training.infer_current_codebase_features(owner.index, owner.repository,
                expected_head=owner.expected_head, registry=registry, version_id=version_id, paths=paths, **controls())
            retained.update(record=record, checkpoint=checkpoint, lineage=lineage, inference=inference,
                metrics=_metrics(checkpoint))
        artifacts = {role: _artifact(role, _wire(value)) for role, value in retained.items()}
        _require(sum(row["size_bytes"] for row in artifacts.values()) <= MAX_RETAINED_BYTES, "retained native context exceeds bound")
        binding = _binding(mode, owner.expected_head, structural, artifacts, record, checkpoint, actual)
        _invocation_binding(binding, retained, record, checkpoint, registry)
        for role, value in retained.items():
            remaining()
            if role != "invocation":
                with (output / (role + ".json")).open("xb") as stream:
                    stream.write(_wire(value))
        raw = _wire(binding)
        with (output / "context.json").open("xb") as stream:
            stream.write(raw)
        context = FrozenCodebaseFeatureContext(output, raw)
        _checked_files(context, remaining)
    # Context exit observes native source again; retained bytes need another fence.
    _checked_files(context, final_pending)
    final_pending()
    return context


def verify_current_context(owner, registry, context):
    """Reobserve source, replay history and fresh inference; never fit or write."""
    from ipfs_datasets_py.logic.software_contracts import codebase_source_training as training
    _require(type(owner) is RepositoryPlanPreviewOwner, "exact native preview owner required")
    deadline = time.monotonic() + owner.timeout_seconds
    _require(type(context) is FrozenCodebaseFeatureContext, "immutable native feature context required")
    binding = context.material_binding
    _require(binding["head"] == owner.expected_head.to_dict(), "context belongs to another selected source head")
    with _current(owner, registry, deadline) as (structural, controls, remaining, final_pending):
        retained = _checked_files(context, remaining)
        record = checkpoint = None
        if context.mode != "model_off":
            _require(owner.memory_mb >= 1024, "native inference memory reservation required")
            record, checkpoint, lineage = _model(owner, registry, context.version_id)
            _require(_wire(record) == _wire(retained["record"]) and _wire(checkpoint) == _wire(retained["checkpoint"])
                and _wire(lineage) == _wire(retained["lineage"]), "retained model differs from native historical replay")
            paths = [row["path"] for row in checkpoint["report"]["codebase_provenance"]["selections"]]
            replay = training.infer_current_codebase_features(owner.index, owner.repository,
                expected_head=owner.expected_head, registry=registry, version_id=context.version_id, paths=paths, **controls())
            for key in ("schema", "head", "version_id", "inference", "training_executed", "authority"):
                _require(_wire(replay[key]) == _wire(retained["inference"][key]), "retained inference differs from fresh native replay")
            receipt = retained["inference"]["worker_receipt"]
            fresh_receipt = replay["worker_receipt"]
            _require(type(receipt) is dict and set(receipt) == set(fresh_receipt), "retained inference receipt fields differ")
            for key in ("executable_sha256", "worker_sha256", "output_sha256", "returncode",
                        "workspace_cleaned", "memory_enforcement", "source_execution_attested"):
                _require(_wire(receipt[key]) == _wire(fresh_receipt[key]), "retained inference receipt differs from current native producer")
            _require(type(receipt["elapsed_ms"]) is int and receipt["elapsed_ms"] >= 0
                and type(receipt["input_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", receipt["input_sha256"])
                and type(receipt["limits"]) is dict and set(receipt["limits"]) == {"resident_memory_bytes", "max_output_bytes"}
                and type(receipt["limits"]["resident_memory_bytes"]) is int
                and 1024 * 1024**2 <= receipt["limits"]["resident_memory_bytes"] <= 4096 * 1024**2
                and type(receipt["limits"]["max_output_bytes"]) is int
                and receipt["limits"]["max_output_bytes"] == training.CodebaseFeatureTrainingLimits().max_candidate_bytes,
                "retained inference execution bounds differ")
        _invocation_binding(binding, retained, record, checkpoint, registry)
        expected = _binding(context.mode, owner.expected_head, structural, binding["artifacts"], record, checkpoint,
                            0 if context.mode != "train" else checkpoint["report"]["attempted_epochs"])
        _require(_wire(expected) == context._material_bytes, "retained material differs from native source/model identity")
        _checked_files(context, remaining)
    _checked_files(context, final_pending)
    result = context.material_binding
    final_pending()
    return result


__all__ = ["SCHEMA", "PROFILE", "CodebaseFeatureContextError", "FrozenCodebaseFeatureContext",
           "prepare_codebase_feature_context", "verify_current_context"]
