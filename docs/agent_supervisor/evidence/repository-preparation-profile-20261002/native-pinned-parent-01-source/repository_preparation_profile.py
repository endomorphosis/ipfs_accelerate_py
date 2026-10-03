"""Explicit finite qualification profiles and source/model admission bindings.

This consumer reuses native datasets owners. The scalar cohort belongs only to
the authored qualification fixture; it is never added to a benchmark checkout.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from .repository_benchmark_preparation import (
    PreparationBudget, RepositoryPreparationSelection, SourceModelSelection,
)
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import CodebaseScanPolicy
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

SCHEMA = "finite-repository-preparation-profile@1"
FROZEN_SCHEMA = "repository-preparation-admission-binding@1"
FIXTURES = {"finite-offset-base@1", "finite-offset-scalar-cohort@1"}


def _require(value, message):
    if not value:
        raise ValueError(message)


def _raw(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _unique(pairs):
    result = {}
    for key, value in pairs:
        _require(key not in result, "duplicate profile field")
        result[key] = value
    return result


@dataclass(frozen=True)
class Source384CheckpointSelection:
    checkpoint_path: str
    checkpoint_sha256: str
    embedding_snapshot: str

    def __post_init__(self):
        _require(type(self.checkpoint_sha256) is str and len(self.checkpoint_sha256) == 64
            and all(c in "0123456789abcdef" for c in self.checkpoint_sha256), "exact checkpoint hash required")
        for name in ("checkpoint_path", "embedding_snapshot"):
            value = getattr(self, name)
            _require(type(value) is str and 0 < len(value) <= 4096, "bounded exact model asset path required")
            path = Path(value)
            _require(path.is_absolute() and str(path) == value and path.resolve() == path
                and not any(p.is_symlink() for p in (path, *path.parents)), "canonical model asset location required")


@dataclass(frozen=True)
class RepositoryPreparationProfile:
    selection: RepositoryPreparationSelection
    budget: PreparationBudget
    checkpoint: Source384CheckpointSelection | None = None
    fixture: str = "finite-offset-base@1"

    def __post_init__(self):
        _require(type(self.selection) is RepositoryPreparationSelection and type(self.budget) is PreparationBudget,
            "native preparation selection and budget required")
        _require(type(self.fixture) is str and self.fixture in FIXTURES, "unknown authored qualification fixture")
        _require((self.selection.model_policy == "model_off" and self.checkpoint is None) or
            (self.selection.model_policy != "model_off" and type(self.checkpoint) is Source384CheckpointSelection),
            "explicit checkpoint must match selected model policy")
        _require(self.budget.structural_memory_mb == 1024, "managed structural phases must reserve1024MiB")
        if self.checkpoint is not None:
            _require(self.budget.memory_mb == 4096, "declared source384 numerical profile requires4096MiB")
        _require(len(self.selection.proof_contracts) <= self.budget.max_proof_contracts,
            "declared proof population exceeds budget")
        _require(self.selection.index_contracts is not None, "profile must declare index contracts independently")

    def to_dict(self):
        return dict(schema=SCHEMA, selection=self.selection.to_dict(), budget=asdict(self.budget),
            checkpoint=None if self.checkpoint is None else asdict(self.checkpoint), fixture=self.fixture)

    @classmethod
    def from_dict(cls, value):
        _require(type(value) is dict and set(value) == {"schema", "selection", "budget", "checkpoint", "fixture"}
            and value["schema"] == SCHEMA, "closed preparation profile required")
        selected = value["selection"]
        _require(type(selected) is dict and set(selected) == {"inventory_policy", "index_contracts", "proof_contracts", "proof_inputs",
            "training_selections", "inference_paths", "model_policy"}, "closed preparation selection required")
        _require(type(value["budget"]) is dict and set(value["budget"]) == set(asdict(PreparationBudget())),
            "closed phase budget required")
        for key in ("index_contracts", "proof_contracts", "proof_inputs", "training_selections", "inference_paths"):
            _require(type(selected[key]) is list, "explicit selection lists required")
        for row in selected["training_selections"]:
            _require(type(row) is dict and set(row) == {"path", "role", "group_id"}, "closed training selection required")
        checkpoint = value["checkpoint"]
        if checkpoint is not None:
            _require(type(checkpoint) is dict and set(checkpoint) == {"checkpoint_path", "checkpoint_sha256", "embedding_snapshot"},
                "closed pinned checkpoint selection required")
            checkpoint = Source384CheckpointSelection(**checkpoint)
        selection = RepositoryPreparationSelection(CodebaseScanPolicy.from_dict(selected["inventory_policy"]),
            proof_contracts=tuple(IntegerOffsetContract.from_dict(v) for v in selected["proof_contracts"]),
            proof_inputs=tuple(tuple(v) for v in selected["proof_inputs"]),
            training_selections=tuple((v["path"], v["role"], v["group_id"]) for v in selected["training_selections"]),
            inference_paths=tuple(selected["inference_paths"]), model_policy=selected["model_policy"],
            index_contracts=tuple(IntegerOffsetContract.from_dict(v) for v in selected["index_contracts"]))
        result = cls(selection, PreparationBudget(**value["budget"]), checkpoint, value["fixture"])
        _require(_raw(result.to_dict()) == _raw(value), "noncanonical preparation profile")
        return result


def load_profile(path):
    path = Path(path)
    _require(path.is_absolute() and path.resolve(strict=True) == path and path.is_file()
        and not any(p.is_symlink() for p in (path, *path.parents)), "existing exact profile file required")
    _require(0 < path.stat().st_size <= 65536, "bounded preparation profile required")
    return RepositoryPreparationProfile.from_dict(json.loads(path.read_bytes(), object_pairs_hook=_unique))


def scalar_cohort():
    """Authored nine-source fixture with disjoint operator roles, not benchmark data."""
    files, selections = {}, []
    for role, operators in (("train", ("+", "-", "*")), ("validation", ("<", "<=", ">")),
                            ("holdout", (">=", "==", "!="))):
        for ordinal, operator in enumerate(operators):
            path = f"cohort/{role}_{ordinal}.py"
            sort = "int" if role == "train" else "bool"
            files[path] = (f"def calculate(capacity: int, threshold: int) -> {sort}:\n"
                           f"    return capacity {operator} threshold\n")
            selections.append((path, role, path))
    return files, tuple(selections)


def register_model(profile, registry):
    """Consume exact shared weights through the native datasets registry only."""
    from ipfs_datasets_py.logic.software_contracts import codebase_source_384 as numerical
    from ipfs_datasets_py.duckdb_control.autoencoder_registry import RUN_LIFECYCLE_SCHEMA
    _require(profile.checkpoint is not None, "model profile required")
    selected = profile.checkpoint
    parent = numerical.register_shared_parent(registry, checkpoint_path=Path(selected.checkpoint_path),
        expected_sha256=selected.checkpoint_sha256)
    variant = registry.get_version(parent)["variant_id"]
    registry.initialize_head("qualification-initialize-parent", variant, "main", parent)
    head = registry.resolve_head(variant, "main")
    return SourceModelSelection(registry, parent, "main", head, selected.embedding_snapshot,
        lifecycle_policy=dict(schema=RUN_LIFECYCLE_SCHEMA, max_attempts=1,
            wall_time_seconds=profile.budget.phase_seconds, memory_bytes=profile.budget.memory_mb*1024**2,
            max_input_bytes=32*1024**2, max_samples=128, optimizer_steps=0, head_refits=1,
            max_checkpoint_bytes=32*1024**2, expected_head=head))


def freeze_preparation(*, index, repository, expected_head, report, selection, model=None, **controls):
    """Recheck current owners and bind immutable preparation before task admission."""
    from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy_live import verify_policy_current
    from ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest import load_codebase_semantic_manifest
    _require(report["qualified"] is True and report["selection"] == selection.to_dict()
        and report["source_head"] == expected_head.to_dict(), "qualified exact preparation required")
    body = {key: value for key, value in report.items() if key != "preparation_cid"}
    raw = _raw(body)
    _require(report["preparation_cid"] == cid_for_structured(dict(schema="repository-preparation-report-bytes@1",
        sha256=_sha(raw), size_bytes=len(raw))), "preparation report changed")
    verify_policy_current(index, repository, expected_head=expected_head,
        receipt_cid=report["policy_receipt_cid"], **controls)
    semantic = load_codebase_semantic_manifest(index, report["semantic_manifest_cid"])
    _require(semantic["source_head"] == expected_head.to_dict(), "prepared semantic source changed")
    _require(report["complete_inventory"] == semantic["coverage"], "complete inventory differs from native manifest")
    model_binding = deepcopy(report["model"])
    if model is None:
        _require(model_binding == {"enabled": False, "identity": "explicit-model-off@1"}, "unexpected selected model")
    else:
        _require(model.registry.resolve_head(model.expected_head["variant_id"], model.branch) == model.expected_head,
            "selected parent head changed before admission")
        version = model.registry.get_version(model_binding["version_id"])
        model.registry.verify_artifact(version["artifact"])
        model_binding.update(selected_version=version, registry_expected_head=deepcopy(model.expected_head),
            inference_sha256=_sha(_raw(report["inference"])))
        _require(report["inference"]["source_head"] == expected_head.to_dict()
            and report["inference"]["version_id"] == version["version_id"], "inference source or model differs")
    binding = dict(schema=FROZEN_SCHEMA, source_head=expected_head.to_dict(),
        semantic_manifest_cid=report["semantic_manifest_cid"], policy_receipt_cid=report["policy_receipt_cid"],
        preparation_cid=report["preparation_cid"], selection=selection.to_dict(), model=model_binding,
        complete_inventory=deepcopy(report["complete_inventory"]),
        model_outputs_are_advisory=True, model_output_used_as_proof=False, planning_strategy="native-finite-symbolic@1",
        execution_authority=False, completion_authority=False)
    binding["binding_cid"] = cid_for_structured(binding)
    return binding
