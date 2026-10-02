"""Explicit, budgeted repository preparation for controlled benchmark arms.

Inventory, training and proof populations are declared independently. This
module prepares evidence; it cannot authorize benchmark solutions or task edits.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import time

from ipfs_datasets_py.duckdb_control.codebase_catalog import CodebaseHead
from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy import (
    CodebaseScanPolicy, prepare_policy_current,
)
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy_live import verify_policy_current
from ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest import build_codebase_semantic_manifest
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from ipfs_accelerate_py.agent_supervisor.runtime.repository_resource_bridge import (
    RepositoryHostReservation, RepositoryPhaseDemand,
)
from ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache import FiniteCheckedCache

SCHEMA = "repository-benchmark-preparation@1"
MODEL_POLICIES = {"model_off", "pinned_parent", "optional_training", "required_training"}
PHASES = {"scan", "semantic_index", "sql", "training", "inference", "proof", "validation"}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class PreparationBudget:
    """Per-operation limits within the separately admitted full trial envelope."""
    phase_seconds: float = 90.
    memory_mb: int = 4096
    max_proof_contracts: int = 8

    def __post_init__(self):
        _require(type(self.phase_seconds) in (int, float) and math.isfinite(self.phase_seconds)
            and 0 < self.phase_seconds <= 180, "bounded phase deadline required")
        _require(type(self.memory_mb) is int and 1024 <= self.memory_mb <= 16384,
                 "bounded phase memory of at least1024MiB required")
        _require(type(self.max_proof_contracts) is int and 0 <= self.max_proof_contracts <= 16,
                 "bounded proof population required")


@dataclass(frozen=True)
class RepositoryPreparationSelection:
    """Immutable declarations, distinct from any inferred property or label."""
    inventory_policy: CodebaseScanPolicy
    proof_contracts: tuple = ()
    proof_inputs: tuple = ()
    training_selections: tuple = ()
    inference_paths: tuple = ()
    model_policy: str = "model_off"

    def __post_init__(self):
        _require(type(self.inventory_policy) is CodebaseScanPolicy and self.model_policy in MODEL_POLICIES,
                 "native inventory policy and explicit model policy required")
        _require(type(self.proof_contracts) is tuple and all(type(v) is IntegerOffsetContract for v in self.proof_contracts)
            and len({v.path for v in self.proof_contracts}) == len(self.proof_contracts),
            "one explicit native proof contract per selected path required")
        _require(type(self.proof_inputs) is tuple and len(self.proof_inputs) == len(self.proof_contracts),
                 "an independent finite input domain for every proof contract is required")
        for domain in self.proof_inputs:
            _require(type(domain) is tuple and 1 <= len(domain) <= 32 and list(domain) == sorted(set(domain))
                and all(type(v) is int and abs(v) <= 2**31 for v in domain), "exact nonempty finite domain required")
        _require(type(self.training_selections) is tuple and len(self.training_selections) <= 128,
                 "bounded immutable training selections required")
        for row in self.training_selections:
            _require(type(row) is tuple and len(row) == 3 and all(type(v) is str and v for v in row)
                and row[1] in {"train", "validation", "holdout"}, "training path, role and group required")
        _require(len({r[0] for r in self.training_selections}) == len(self.training_selections),
                 "duplicate training source selection")
        _require(type(self.inference_paths) is tuple and len(self.inference_paths) <= 128
            and all(type(v) is str and v for v in self.inference_paths)
            and len(set(self.inference_paths)) == len(self.inference_paths), "explicit distinct inference paths required")
        if self.model_policy == "model_off":
            _require(not self.training_selections and not self.inference_paths,
                     "model-off arm cannot silently train or infer")
        else:
            _require(bool(self.inference_paths), "model-selected arm needs an explicit inference population")
        if self.model_policy in {"optional_training", "required_training"}:
            _require(len(self.training_selections) >= 3, "training requires a declared split cohort")
        else:
            _require(not self.training_selections, "non-training arm cannot admit training samples")

    def to_dict(self):
        return dict(inventory_policy=self.inventory_policy.to_dict(),
            proof_contracts=[v.to_dict() for v in self.proof_contracts],
            proof_inputs=[list(v) for v in self.proof_inputs],
            training_selections=[dict(path=p, role=r, group_id=g) for p,r,g in self.training_selections],
            inference_paths=list(self.inference_paths), model_policy=self.model_policy)


@dataclass(frozen=True)
class SourceModelSelection:
    """Ephemeral existing model owner; identity is independently checked by it."""
    registry: object
    parent_version_id: str
    branch: str
    expected_head: dict
    embedding_snapshot: str
    lifecycle_policy: dict | None = None


class RequiredTrainingQualificationError(ValueError):
    def __init__(self, report):
        self.report = deepcopy(report)
        super().__init__("required repository training did not pass the parent retention gate")


def _retained(evaluation):
    parent, child = evaluation["parent_holdout"], evaluation["child_holdout"]
    return (parent["count"] > 0 and parent["count"] == child["count"]
        and child["valid_candidates"] == child["count"]
        and child["exact_targets"] >= parent["exact_targets"])


def prepare_repository_benchmark(*, index, repository, repository_id, expected_head,
        operation_id, selection, envelope, budget=PreparationBudget(), checked_cache=None,
        tool_policy=None, model=None):
    """Run selected native stages under the actual shared host parent.

    No fit occurs for model_off/pinned_parent. Required-training failure has no
    fallback. Optional-training retains a pinned parent after a failed child.
    No child is promoted automatically. Returned model outputs remain proposals.
    """
    _require(type(selection) is RepositoryPreparationSelection and type(budget) is PreparationBudget
        and type(envelope) is RepositoryHostReservation, "typed selection, budget and live shared host envelope required")
    _require(envelope.repository_id == repository_id and len(selection.proof_contracts) <= budget.max_proof_contracts,
             "repository identity or proof budget differs")
    _require(selection.model_policy == "model_off" and model is None or
        selection.model_policy != "model_off" and type(model) is SourceModelSelection,
        "explicit model selection must match the arm")
    if selection.proof_contracts:
        _require(type(checked_cache) is FiniteCheckedCache and checked_cache.artifacts is index.artifacts
            and type(tool_policy) is dict, "proof arm requires exact native cache and tool policy")
    else:
        _require(checked_cache is None and tool_policy is None, "unselected proof resources must be absent")
    declared = selection.to_dict()
    frozen_model_head = None if model is None else deepcopy(model.expected_head)
    started = time.monotonic()
    report = dict(schema=SCHEMA, qualified=False, selection=declared, source_head=None,
        policy_receipt_cid=None, semantic_manifest_cid=None, discovery=None,
        training=None, inference=None, model={"enabled":False,"identity":"explicit-model-off@1"},
        checked_proofs=[], stages=[], model_provider_calls=0, model_provider_tokens=0,
        execution_authority=False, completion_authority=False, model_promotion_performed=False,
        evaluation_population_use=dict(training_holdout="development_retention_gate",
            holdout_used_for_runtime_selection=selection.model_policy in {"optional_training", "required_training"},
            untouched_final_benchmark_test=False),
        benchmark_score=False, deadline_enforcement="cooperative_parent_and_native_subprocess_limits")

    @contextmanager
    def phase(name):
        _require(name in PHASES, "unknown phase")
        start = time.monotonic()
        stage = dict(phase=name, status="failed", elapsed_seconds=None)
        report["stages"].append(stage)
        with envelope.phase(RepositoryPhaseDemand(name, memory_mb=budget.memory_mb)) as lease:
            controls = lease.native_options()
            phase_left = budget.phase_seconds - (time.monotonic() - start)
            _require(phase_left > 0, "phase queue exhausted its declared deadline")
            controls["timeout_seconds"] = min(controls["timeout_seconds"], phase_left)
            try:
                yield controls
                envelope.remaining()
                _require(time.monotonic()-start <= budget.phase_seconds, "phase exceeded its declared deadline")
                stage["status"] = "completed"
            finally:
                stage["elapsed_seconds"] = time.monotonic()-start

    with phase("scan") as controls:
        policy = prepare_policy_current(index, repository, repository_id=repository_id,
            operation_id=operation_id, expected_head=expected_head, policy=selection.inventory_policy,
            training_paths=tuple(r[0] for r in selection.training_selections),
            proof_paths=tuple(v.path for v in selection.proof_contracts), **controls)
    head = CodebaseHead.from_dict(policy["head"])
    report.update(source_head=head.to_dict(), policy_receipt_cid=policy["receipt_cid"])
    with phase("semantic_index"):
        semantic = build_codebase_semantic_manifest(index, policy_receipt_cid=policy["receipt_cid"],
            contracts=selection.proof_contracts)
        report.update(semantic_manifest_cid=semantic["manifest_cid"], complete_inventory=semantic["coverage"])
    with phase("sql") as controls:
        catalog = IntentCodebaseCatalog(index)
        report["discovery"] = catalog.publish(repository, expected_head=head,
            manifest_cid=semantic["manifest_cid"], operation_id=operation_id+":discovery", **controls)
    if model is not None:
        from ipfs_datasets_py.logic.software_contracts import codebase_source_384 as numerical
        registry = model.registry
        _require(registry.resolve_head(model.expected_head["variant_id"], model.branch) == model.expected_head
            and model.expected_head["version_id"] == model.parent_version_id, "selected parent model head changed")
        selected_version = model.parent_version_id
        report["model"] = dict(enabled=True, identity="pinned-source384-parent@1", version_id=selected_version,
            expected_head=deepcopy(model.expected_head), fallback_reason=None)
        if selection.model_policy in {"optional_training", "required_training"}:
            from ipfs_datasets_py.logic.software_contracts import codebase_training_corpus as corpus
            from ipfs_datasets_py.logic.software_contracts import codebase_training_lifecycle as lifecycle
            with phase("semantic_index"):
                frozen = corpus.freeze_corpus(index, expected_head=head, selections=declared["training_selections"])
                job = lifecycle.prepare_job(index, repository, expected_head=head, frozen_corpus=frozen,
                    registry=registry, parent_version_id=model.parent_version_id, branch=model.branch,
                    expected_model_head=model.expected_head, operation_id=operation_id+":training",
                    embedding_snapshot=model.embedding_snapshot, policy=model.lifecycle_policy)
            with phase("training") as controls:
                trained = lifecycle.execute_job(job, **controls)
                report["training"] = trained
            if trained["status"] == "completed" and _retained(trained["evaluation"]):
                selected_version = trained["numerical_child_version_id"]
                report["model"].update(identity="qualified-unpromoted-source384-child@1", version_id=selected_version)
            else:
                report["model"]["fallback_reason"] = "child_unqualified_or_incomplete"
                if selection.model_policy == "required_training":
                    raise RequiredTrainingQualificationError(report)
        with phase("inference") as controls:
            if selected_version == model.parent_version_id:
                # Additive parent route is supplied by the registered CodebaseIR
                # runtime, preserving the frozen source384 child contract.
                from ipfs_datasets_py.logic.software_contracts.codebase_runtime_384 import infer_shared_parent
                inferred = infer_shared_parent(index, repository, expected_head=head, registry=registry,
                    version_id=selected_version, paths=selection.inference_paths,
                    embedding_snapshot=model.embedding_snapshot, **controls)
            else:
                inferred = numerical.infer_current_source384(index, repository, expected_head=head,
                    registry=registry, version_id=selected_version, paths=selection.inference_paths,
                    embedding_snapshot=model.embedding_snapshot, **controls)
            report["inference"] = inferred
    for contract, inputs in zip(selection.proof_contracts, selection.proof_inputs):
        with phase("proof") as controls:
            timeout = controls.pop("timeout_seconds")
            # The proof owner has its own fixed checked-worker memory bounds.
            controls.pop("memory_mb")
            result = checked_cache.check_and_store(owner_inputs=dict(index=index, repository=repository,
                expected_head=head, contract=contract, inputs=list(inputs), tool_policy=tool_policy,
                **controls), timeout_seconds=timeout)
            report["checked_proofs"].append({k:result[k] for k in ("record_cid","request_key","status","scope","positive_reuse_eligible")})
    with phase("validation") as controls:
        verify_policy_current(index, repository, expected_head=head, receipt_cid=policy["receipt_cid"], **controls)
        _require(selection.to_dict() == declared, "preparation selection changed")
        if model is not None:
            _require(model.expected_head == frozen_model_head and
                model.registry.resolve_head(frozen_model_head["variant_id"],model.branch) == frozen_model_head,
                     "selected model head changed during preparation")
    report.update(qualified=True, elapsed_seconds=time.monotonic()-started, resource_receipt=envelope.receipt())
    # Native scheduler timestamps are measured floats. Commit their exact
    # bounded JSON bytes instead of coercing them into semantic DAG-JSON values.
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    report["preparation_cid"] = cid_for_structured(dict(schema="repository-preparation-report-bytes@1",
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
    return report
