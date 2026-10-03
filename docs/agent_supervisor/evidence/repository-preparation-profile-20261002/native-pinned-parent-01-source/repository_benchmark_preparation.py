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
from pathlib import Path
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
    structural_memory_mb: int | None = None

    def __post_init__(self):
        _require(type(self.phase_seconds) in (int, float) and math.isfinite(self.phase_seconds)
            and 0 < self.phase_seconds <= 180, "bounded phase deadline required")
        _require(type(self.memory_mb) is int and 1024 <= self.memory_mb <= 16384,
                 "bounded phase memory of at least1024MiB required")
        _require(type(self.max_proof_contracts) is int and 0 <= self.max_proof_contracts <= 16,
                 "bounded proof population required")
        _require(self.structural_memory_mb is None or (type(self.structural_memory_mb) is int
            and 1024 <= self.structural_memory_mb <= self.memory_mb),
            "structural memory must be an explicit bounded subset of the phase memory")

    def memory_for(self, phase):
        return self.memory_mb if phase in {"training", "inference"} else (
            self.memory_mb if self.structural_memory_mb is None else self.structural_memory_mb)


@dataclass(frozen=True)
class RepositoryPreparationSelection:
    """Immutable declarations, distinct from any inferred property or label."""
    inventory_policy: CodebaseScanPolicy
    proof_contracts: tuple = ()
    proof_inputs: tuple = ()
    training_selections: tuple = ()
    inference_paths: tuple = ()
    model_policy: str = "model_off"
    index_contracts: tuple | None = None

    def __post_init__(self):
        _require(type(self.inventory_policy) is CodebaseScanPolicy and self.model_policy in MODEL_POLICIES,
                 "native inventory policy and explicit model policy required")
        _require(type(self.proof_contracts) is tuple and all(type(v) is IntegerOffsetContract for v in self.proof_contracts)
            and len({v.path for v in self.proof_contracts}) == len(self.proof_contracts),
            "one explicit native proof contract per selected path required")
        _require(self.index_contracts is None or (type(self.index_contracts) is tuple
            and all(type(v) is IntegerOffsetContract for v in self.index_contracts)
            and len({v.path for v in self.index_contracts}) == len(self.index_contracts)),
            "explicit native index declarations must be distinct from proof selection")
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
            index_contracts=None if self.index_contracts is None else [v.to_dict() for v in self.index_contracts],
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
        tool_policy=None, model=None, pipeline=None, pipeline_attempt_root=None,
        pipeline_write_budget_bytes=16 * 1024 * 1024):
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
    if pipeline is None:
        _require(pipeline_attempt_root is None, "pipeline attempt root requires an explicit managed pipeline")
    else:
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_pipeline_resources import RepositoryPipelineReservation
        _require(type(pipeline) is RepositoryPipelineReservation and pipeline.parent is envelope,
                 "pipeline must own this exact admitted host envelope")
        _require(type(pipeline_write_budget_bytes) is int and 0 < pipeline_write_budget_bytes <= 1024**3,
                 "bounded exact managed write ceiling required")
        _require(pipeline_attempt_root is not None, "managed preparation needs an explicit attempt root")
        pipeline_attempt_root = Path(pipeline_attempt_root).absolute()
        _require(pipeline_attempt_root.is_dir(), "existing managed attempt root required")
        outputs = [pipeline_attempt_root, index.artifacts.root,
                   getattr(index.catalog, "_database_path", None)]
        if checked_cache is not None:
            outputs.append(checked_cache.cache.path)
        if model is not None:
            outputs.extend((model.registry.database_path, model.registry.artifact_root))
        for output in outputs:
            _require(output is not None, "managed preparation needs durable native output owners")
            output = Path(output).absolute()
            _require(not any(part.is_symlink() for part in (output, *output.parents))
                and any(output == root or root in output.parents for root in pipeline.roots),
                "preparation output owner is outside the exact named disk roots")
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
    if pipeline is not None:
        report["managed_pipeline"] = dict(schema="repository-managed-preparation@1",
            external_write_ceiling_bytes_per_phase=pipeline_write_budget_bytes,
            disk_enforcement="precharged_ceiling_plus_sampled_final_named_root_growth",
            external_process_rss_captured=False, hard_disk_quota=False)
    training_job = None

    @contextmanager
    def phase_work(name):
        _require(name in PHASES, "unknown phase")
        start = time.monotonic()
        stage = dict(phase=name, status="failed", elapsed_seconds=None)
        report["stages"].append(stage)
        memory_mb = budget.memory_for(name)
        if pipeline is None:
            context = envelope.phase(RepositoryPhaseDemand(name, memory_mb=memory_mb))
        else:
            request = json.dumps(dict(schema="repository-preparation-phase-request@1",
                repository_id=repository_id, operation_id=operation_id, phase=name,
                selection=declared), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            attempt = pipeline_attempt_root / (hashlib.sha256(operation_id.encode()).hexdigest()[:24]
                + "-" + str(len(report["stages"])) + "-" + name)
            attempt.mkdir(mode=0o700)
            context = pipeline.phase(RepositoryPhaseDemand(name, memory_mb=memory_mb,
                disk_bytes=pipeline_write_budget_bytes), payload=request, attempt_directory=attempt)
        with context as lease:
            controls = lease.native_options()
            phase_left = budget.phase_seconds - (time.monotonic() - start)
            _require(phase_left > 0, "phase queue exhausted its declared deadline")
            controls["timeout_seconds"] = min(controls["timeout_seconds"], phase_left)
            before = None
            if pipeline is not None:
                before = lease.check_usage()["observed_apparent_bytes"]
                # Shared CAS/SQL output owners remain in their native locations.
                # Retain this conservative full precharge, even when actual
                # final growth is smaller; do not invent per-phase byte savings.
                lease.charge_external(pipeline.roots[0], pipeline_write_budget_bytes)
            try:
                yield controls
                envelope.remaining()
                _require(time.monotonic()-start <= budget.phase_seconds, "phase exceeded its declared deadline")
                if pipeline is not None:
                    after = lease.check_usage()["observed_apparent_bytes"]
                    growth = max(0, after-before)
                    _require(growth <= pipeline_write_budget_bytes,
                             "sampled named-root growth exceeds managed write ceiling")
                    lease.finalize(artifacts_durable=True)
                    stage["managed_resources"] = dict(payload_bytes=len(request),
                        external_write_ceiling_bytes=pipeline_write_budget_bytes,
                        sampled_named_root_growth_bytes=growth, named_roots_only=True,
                        disk_reservation_id=lease.daemon.reservation_id)
                stage["status"] = "completed"
            finally:
                stage["elapsed_seconds"] = time.monotonic()-start

    @contextmanager
    def phase(name):
        """Retain queue, consumer and cleanup failures with the original type."""
        start = time.monotonic()
        position = len(report["stages"])
        try:
            with phase_work(name) as controls:
                yield controls
        except BaseException as error:
            if len(report["stages"]) > position:
                report["stages"][position].update(status="failed",
                    elapsed_seconds=time.monotonic()-start, error_type=type(error).__name__)
            report["elapsed_seconds"] = time.monotonic()-started
            if name == "training" and training_job is not None:
                try:
                    report["training_state_on_failure"] = lifecycle.inspect_job(training_job)
                except Exception as observation_error:
                    report["training_state_observation_error"] = type(observation_error).__name__
            error.preparation_report = deepcopy(report)
            raise

    with phase("scan") as controls:
        policy = prepare_policy_current(index, repository, repository_id=repository_id,
            operation_id=operation_id, expected_head=expected_head, policy=selection.inventory_policy,
            training_paths=tuple(r[0] for r in selection.training_selections),
            proof_paths=tuple(v.path for v in selection.proof_contracts), **controls)
    head = CodebaseHead.from_dict(policy["head"])
    report.update(source_head=head.to_dict(), policy_receipt_cid=policy["receipt_cid"])
    with phase("semantic_index"):
        semantic = build_codebase_semantic_manifest(index, policy_receipt_cid=policy["receipt_cid"],
            contracts=selection.proof_contracts if selection.index_contracts is None else selection.index_contracts)
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
                training_job = job
                report["training_job"] = job.to_dict()
            with phase("training") as controls:
                report["numerical_training_attempted"] = True
                trained = lifecycle.execute_job(job, **controls)
                report["training"] = trained
            if trained["status"] == "completed" and _retained(trained["evaluation"]):
                selected_version = trained["numerical_child_version_id"]
                report["model"].update(identity="qualified-unpromoted-source384-child@1", version_id=selected_version)
            else:
                report["model"]["fallback_reason"] = "child_unqualified_or_incomplete"
                if selection.model_policy == "required_training":
                    report["elapsed_seconds"] = time.monotonic()-started
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
    if pipeline is not None:
        report["managed_pipeline"]["resource_receipt"] = pipeline.receipt()
    # Native scheduler timestamps are measured floats. Commit their exact
    # bounded JSON bytes instead of coercing them into semantic DAG-JSON values.
    raw = json.dumps(report, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    report["preparation_cid"] = cid_for_structured(dict(schema="repository-preparation-report-bytes@1",
        sha256=hashlib.sha256(raw).hexdigest(), size_bytes=len(raw)))
    return report
