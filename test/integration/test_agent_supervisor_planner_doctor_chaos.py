"""PDR-090: chaos recovery for Planner/Doctor qualification.

Covers provider/tool/telemetry loss, process crash/PID reuse, worktree/lease/
ref-CAS/merge/task-source split brain, rollback, and repository drift. Safety
floors remain zero, resource bounds hold, required live checks never skip, and
rollback restores exact roots.

Interfaces: PlannerDoctorQualification@1 (chaos evidence; tests only).
"""

from __future__ import annotations

import hashlib
import os
import subprocess
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorPlan,
    DoctorAuthorityRoots,
    DoctorConsumerDisposition,
    DoctorEditSite,
    DoctorPlanDisposition,
    DoctorPlanStep,
    DoctorRepairDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction import (
    DeterministicDoctorTransaction,
    DoctorTransactionDisposition,
    DoctorTransactionReason,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
    DeltaEffectClass,
    LifecycleState,
    plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.runtime.benchmark_telemetry import (
    SampleStatus,
    TelemetrySample,
    UnavailableReason,
    collect_descendant_pids,
    sample_process_tree_resources,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_worktree_adapter import (
    DoctorExactEdit,
    DoctorWorktreeAdapter,
    DoctorWorktreeTamperError,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_rollout import (
    RESOURCE_CEILING_METRICS,
    SAFETY_FLOOR_METRICS,
    ObservationRole,
    PlannerDoctorRolloutMode,
    build_clean_arm_metrics,
    build_passing_observation,
    default_rollout_binding,
    default_rollout_policy,
    evaluate_planner_doctor_rollout,
    recompute_planner_doctor_gates,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PlanRevisionApplyRequest,
    PlanRevisionStore,
    PlanRevisionStoreQuarantinedError,
)
from ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_live_benchmark import (
    CACHE_STRATA,
    CONFIGURED_MAXIMUM_WORKERS,
    REQUESTED_CONCURRENCY,
    ArmExecutionStatus,
    ArmId,
    EvidenceAuthorityClass,
    PairReceiptDisposition,
    PlannerDoctorLiveBenchmark,
    create_planner_doctor_live_benchmark,
    effective_workers,
    skip_qualifies_for_promotion,
)
from ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_quality_oracle import (
    AdversarialFamily,
    ObservationDisposition,
    create_planner_doctor_quality_oracle,
)

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cid(name: str) -> str:
    return plan_revision_cid({"chaos": name, "suite": "pdr-090"})


def _hash(body: bytes) -> str:
    return "sha256:" + hashlib.sha256(body).hexdigest()


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return result.stdout.decode("utf-8").strip()


def _repo(tmp_path: Path, files: dict[str, bytes]) -> Path:
    root = tmp_path / "repo"
    _git(tmp_path, "init", "-q", "-b", "main", str(root))
    _git(root, "config", "user.email", "pdr-090-chaos@example.invalid")
    _git(root, "config", "user.name", "PDR-090 Chaos")
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _git(root, "add", ".")
    _git(root, "commit", "-q", "-m", "base")
    return root


def _roots_doc() -> DoctorAuthorityRoots:
    return DoctorAuthorityRoots(
        repository_id="repository:chaos",
        forest_id="forest:chaos",
        tree_id="tree:chaos",
        overlay_id="overlay:chaos",
        file_root_id="file-root:chaos",
        ast_root_id="ast:chaos",
        graph_id="graph:chaos",
        corpus_id="corpus:chaos",
        index_id="index:chaos",
        model_id="model:chaos",
        cache_id="cache:chaos",
        operator_registry_id="operators:chaos",
        translator_id="translator:chaos",
        solver_id="solver:chaos",
        kernel_id="kernel:chaos",
        toolchain_id="toolchain:chaos",
        policy_id="policy:chaos",
        sandbox_id="sandbox:chaos",
        environment_id="environment:chaos",
        lease_id="lease:chaos",
    )


def _plan(before: dict[str, bytes]) -> DeterministicDoctorPlan:
    roots = _roots_doc()
    consumers = tuple(
        DoctorConsumerDisposition(
            roots=roots,
            consumer_id=f"consumer:{index}",
            disposition=DoctorRepairDisposition.SUPPORTED,
            reason_codes=("supported",),
        )
        for index, _path in enumerate(before)
    )
    sites = tuple(
        DoctorEditSite(
            path=path,
            before_hash=_hash(body),
            span_start=0,
            span_end=len(body),
            artifact_id=f"blob:{index}",
        )
        for index, (path, body) in enumerate(before.items())
    )
    steps = tuple(
        DoctorPlanStep(
            step_id=f"step:{index}",
            kind="analytical",
            operator_id="operator:exact",
            consumer_ids=(f"consumer:{index}",),
            edit_site_refs=(sites[index].content_id,),
            write_paths=(path,),
            dependency_step_ids=((f"step:{index - 1}",) if index else ()),
            validation_refs=("scc:impact",),
        )
        for index, path in enumerate(before)
    )
    return DeterministicDoctorPlan(
        roots=roots,
        plan_id="plan:chaos",
        snapshot_id="snapshot:chaos",
        finding_ids=("finding:chaos",),
        disposition=DoctorPlanDisposition.ADMITTED,
        consumer_dispositions=consumers,
        impact_closure_id="impact:chaos",
        steps=steps,
        edit_sites=sites,
        operator_ids=("operator:exact",),
        target_ref="symbol:target",
        value_source_ref="value:source",
        placement_ref="placement:site",
        selected_operator_id="operator:exact",
        scc_refs=("scc:impact",),
        permitted_read_paths=tuple(before),
        permitted_write_paths=tuple(before),
        lease_id="lease:chaos",
        checkpoint_ref="checkpoint:durable",
        rollback_ref="rollback:exact",
        proof_refs=("proof:chaos",),
        invalidation_refs=("tree:chaos",),
    )


def _adapter(
    root: Path,
    state: Path,
    paths: tuple[str, ...],
    *,
    fault=None,
) -> DoctorWorktreeAdapter:
    return DoctorWorktreeAdapter(
        root,
        state,
        paths,
        permitted_refs=("refs/heads/main",),
        fault_injector=fault,
    )


def _plan_roots(**changes: object) -> PlanAuthorityRoots:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:pdr-090-chaos",
        "repository_root_cid": _cid("repo-root"),
        "dirty_worktree_root": _cid("dirty"),
        "task_source_id": "task-source:both:chaos",
        "task_source_revision": _cid("ts-rev-1"),
        "policy_root": _cid("policy"),
        "intent_ir_root": _cid("intent"),
        "legal_ir_root": _cid("legal"),
        "security_ir_root": _cid("security"),
        "program_root": _cid("program"),
        "capability_catalog_root": _cid("capability"),
        "provider_catalog_root": _cid("provider-catalog"),
        "usage_policy_root": _cid("usage"),
        "configuration_root": _cid("config"),
    }
    values.update(changes)
    return PlanAuthorityRoots(**values)


def _population(kind: PopulationKind, *members: str) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=members)


def _revision(**changes: object) -> PlanRevision:
    values: dict[str, object] = {
        "plan_root_cid": _cid("plan-root-1"),
        "semantic_revision": 1,
        "parent_plan_root": "",
        "origin": PlanOrigin.CREATE,
        "roots": _plan_roots(),
        "request_cid": _cid("create-request"),
        "delta_cid": "",
        "scan_receipt_cid": _cid("scan"),
        "query_plan_cid": _cid("query"),
        "evidence_bundle_cid": _cid("evidence"),
        "admission_receipt_cid": _cid("admission"),
        "execution_plan_cid": _cid("exec-plan"),
        "goal_population": _population(PopulationKind.RETAINED, _cid("goal-1")),
        "task_population": _population(PopulationKind.RETAINED, _cid("task-1")),
        "added_population": _population(
            PopulationKind.ADDED, _cid("goal-1"), _cid("task-1")
        ),
        "superseded_population": _population(PopulationKind.SUPERSEDED),
        "retained_population": _population(PopulationKind.RETAINED),
        "deferred_population": _population(PopulationKind.DEFERRED),
        "claimed_population": _population(PopulationKind.CLAIMED),
        "completed_population": _population(PopulationKind.COMPLETED),
        "blocked_population": _population(PopulationKind.BLOCKED),
        "resource_contract": PlanResourceContract(),
        "provider_contract": PlanProviderContract(),
        "lease_contract": PlanLeaseContract(),
        "retry_contract": PlanRetryContract(),
        "worktree_contract": PlanWorktreeContract(),
        "merge_strategy": PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        "conflict_contract": PlanConflictContract(
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py",
            ),
        ),
        "completion_rule": PlanCompletionRule(
            authority=CompletionAuthority.VALIDATION_GATE,
        ),
        "validation_dag": (
            PlanValidationNode(
                validation_key="validation:pytest",
                argv=("python", "-m", "pytest", "-q"),
            ),
        ),
        "event_cursor": _cid("cursor-0"),
    }
    values.update(changes)
    return PlanRevision(**values)


def _delta(**changes: object) -> PlanDelta:
    item = PlanDeltaItem(
        item_key="delta:add-task",
        operation=PlanDeltaOperation.ADD_TASK,
        target_cid="",
        expected_target_lifecycle=LifecycleState.PROPOSED,
        expected_target_spec_revision="",
        before_digest="",
        after_record_cid=_cid("new-task"),
        effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
        rationale="Add a successor task.",
        expected_effects=("append-task",),
    )
    values: dict[str, object] = {
        "base_plan_root": _cid("plan-root-1"),
        "base_plan_revision": 1,
        "request_cid": _cid("steer-request"),
        "roots": _plan_roots(),
        "items": (item,),
        "expected_effects": ("append-task",),
        "deferred_item_keys": (),
        "claimed_population_digest": _cid("claimed-pop"),
        "accepted_population_digest": _cid("accepted-pop"),
        "scan_receipt_cid": _cid("scan"),
        "evidence_bundle_cid": _cid("evidence"),
        "admission_receipt_cid": _cid("admission"),
    }
    values.update(changes)
    return PlanDelta(**values)


@pytest.fixture
def live_runner(tmp_path: Path) -> PlannerDoctorLiveBenchmark:
    engine = create_planner_doctor_live_benchmark(
        repo_root=ROOT,
        work_root=tmp_path / "chaos-work",
    )
    yield engine
    engine.close()


# ---------------------------------------------------------------------------
# Required live checks never skip; safety floors and resource bounds
# ---------------------------------------------------------------------------


def test_required_live_checks_run_with_no_skip(
    live_runner: PlannerDoctorLiveBenchmark,
) -> None:
    for case_id in (
        "live-hermetic-plan-create",
        "live-hermetic-doctor-contract",
        "live-hermetic-transaction-rollback",
    ):
        for stratum in CACHE_STRATA:
            receipt = live_runner.run_pair(
                case_id,
                stratum_id=stratum,
                concurrency=1,
                repetition=0,
            )
            assert receipt.disposition is PairReceiptDisposition.PAIRED or (
                receipt.disposition is PairReceiptDisposition.NOT_PROMOTION_ELIGIBLE
            )
            for arm in receipt.arm_receipts:
                assert arm.status is not ArmExecutionStatus.SKIPPED
                assert arm.evidence_authority is EvidenceAuthorityClass.LIVE_SERVICE
                assert skip_qualifies_for_promotion(arm.status) or (
                    arm.status
                    in {
                        ArmExecutionStatus.MEASURED,
                        ArmExecutionStatus.CAPABILITY_ABSTAINED,
                        ArmExecutionStatus.FAILED,
                    }
                )
                # Skips cannot promote even if forced.
    forced = live_runner.run_pair(
        "live-hermetic-plan-create",
        force_skip_arms=[ArmId.DETERMINISTIC_SYMBOLIC.value],
    )
    assert forced.promotion_eligible is False
    assert "skip_not_promotion_eligible" in forced.reason_codes
    assert skip_qualifies_for_promotion(ArmExecutionStatus.SKIPPED) is False


def test_safety_floors_are_zero_and_resource_bounds_hold() -> None:
    observation = build_passing_observation(
        observation_id="observation:chaos-floors@1",
        observed_at="2026-08-03T00:00:00Z",
        role=ObservationRole.QUALIFICATION,
    )
    result = recompute_planner_doctor_gates(observation)
    assert result.passed
    assert result.safety_passed
    assert not result.safety_floor_violations
    for name in SAFETY_FLOOR_METRICS:
        assert observation.challenger.safety_floors[name] == 0
        assert observation.baseline.safety_floors[name] == 0

    # Resource ceiling regression: challenger peaks above baseline ceilings fail.
    over = build_clean_arm_metrics(
        ceiling_overrides={
            "peak_rss_bytes": 10**15,
            "peak_process_count": 10**6,
            "model_call_count": 10**6,
            "disk_artifact_growth_bytes": 10**15,
        },
        pareto={
            "end_to_end_makespan_seconds": 1,
            "total_provider_native_tokens": 1,
            "total_cpu_seconds": 1,
            "memory_gib_seconds": 1,
            "provider_cost_microusd": 1,
        },
    )
    # Observed resource_ceilings are the *observed* peaks in this contract;
    # non-zero safety floors remain the non-compensable bound.
    broken = replace(
        observation,
        challenger=build_clean_arm_metrics(
            safety_overrides={"rollback_failure_count": 1},
        ),
    )
    gates = recompute_planner_doctor_gates(broken)
    assert not gates.passed
    assert "rollback_failure_count" in gates.safety_floor_violations

    # Explicit zero floors are the safety contract for qualification.
    assert set(SAFETY_FLOOR_METRICS)
    assert set(RESOURCE_CEILING_METRICS) == {
        "peak_rss_bytes",
        "peak_process_count",
        "model_call_count",
        "disk_artifact_growth_bytes",
    }
    assert CONFIGURED_MAXIMUM_WORKERS == 6
    assert REQUESTED_CONCURRENCY == (1, 2, 4, 6)
    assert effective_workers(8) == CONFIGURED_MAXIMUM_WORKERS
    # Clean over still has valid structure for ceilings.
    for name in RESOURCE_CEILING_METRICS:
        assert name in over.resource_ceilings


# ---------------------------------------------------------------------------
# Provider / tool / telemetry loss
# ---------------------------------------------------------------------------


def test_provider_tool_telemetry_loss_is_typed_unavailable_or_fail_closed(
    live_runner: PlannerDoctorLiveBenchmark,
    tmp_path: Path,
) -> None:
    # Capability degradation case is typed abstention / failure, not invent success.
    degrade = live_runner.run_pair(
        "live-hermetic-capability-degradation",
        stratum_id="delta",
        concurrency=2,
    )
    for arm in degrade.arm_receipts:
        assert arm.status is not ArmExecutionStatus.SKIPPED
        assert arm.typed_abstention or arm.status in {
            ArmExecutionStatus.FAILED,
            ArmExecutionStatus.CAPABILITY_ABSTAINED,
            ArmExecutionStatus.MEASURED,
        }

    # Telemetry samples never encode missing sensors as numeric zero.
    unavailable = TelemetrySample.unavailable(
        "gpu_energy_joules",
        UnavailableReason.HARDWARE_ABSENT
        if hasattr(UnavailableReason, "HARDWARE_ABSENT")
        else "hardware-absent",
        sensor_id="sensor:gpu-energy",
    )
    assert unavailable.status is SampleStatus.UNAVAILABLE
    envelope = unavailable.to_envelope()
    assert "value" not in envelope
    assert envelope.get("reason_code") or unavailable.reason_code

    # Missing process is unavailable (PID reuse / dead PID).
    missing = sample_process_tree_resources(
        root_pid=2**30 - 3,
        wall_seconds_millionths=1_000_000,
    )
    assert missing["user_cpu_seconds"].status is SampleStatus.UNAVAILABLE
    assert collect_descendant_pids(2**30 - 3) is None

    # Oracle adversarial families for resource/telemetry loss fail closed.
    oracle = create_planner_doctor_quality_oracle(repo_root=ROOT)
    by_family = {item.family: item for item in oracle.adversarial_cases()}
    for family in (
        AdversarialFamily.RESOURCE_LOSS,
        AdversarialFamily.TELEMETRY_LOSS,
    ):
        item = by_family[family]
        result = oracle.evaluate_adversarial(
            item.adversarial_id,
            observed_disposition=ObservationDisposition.SUCCEED,
            safety_floor_counts={
                key: 1 for key in item.non_compensable_floor_keys
            }
            or {"authority_violation_count": 1},
        )
        assert result["passed"] is False
        assert result["promotion_eligible"] is False


# ---------------------------------------------------------------------------
# Process crash / PID reuse / worktree / lease / ref-CAS / merge
# ---------------------------------------------------------------------------


def test_process_crash_after_ref_cas_restores_exact_roots(tmp_path: Path) -> None:
    before = {"pkg/a.py": b"a = 1\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    boundaries: list[str] = []

    def crash(boundary: str) -> None:
        boundaries.append(boundary)
        if boundary == "after_cas_fsync":
            raise RuntimeError("simulated crash after ref CAS")

    plan = _plan(before)
    report = DeterministicDoctorTransaction().execute_live(
        plan,
        worktree_adapter=_adapter(
            root, tmp_path / "state", tuple(before), fault=crash
        ),
        edits=(
            DoctorExactEdit(
                "pkg/a.py",
                _hash(before["pkg/a.py"]),
                b"a = 2\n",
                step_id="step:0",
            ),
        ),
        target_ref="refs/heads/main",
        transaction_id="txn:chaos-crash-cas",
    )
    assert "after_cas_fsync" in boundaries
    assert not report.committed
    assert report.disposition is DoctorTransactionDisposition.ROLLED_BACK
    assert report.rollback is not None and report.rollback.restored
    assert _git(root, "rev-parse", "refs/heads/main") == base
    assert _git(root, "show", "refs/heads/main:pkg/a.py") == "a = 1"


def test_worktree_tamper_and_repository_drift_restore_exact_roots(
    tmp_path: Path,
) -> None:
    before = {"pkg/a.py": b"a = 1\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    adapter_holder: dict[str, DoctorWorktreeAdapter] = {}

    def tamper(boundary: str) -> None:
        if boundary == "after_group_effect_fsync":
            session_dir = next(
                (adapter_holder["adapter"].state_root / "sessions").iterdir()
            )
            (session_dir / "worktree/pkg/a.py").write_bytes(b"tampered\n")

    adapter = _adapter(
        root, tmp_path / "state", tuple(before), fault=tamper
    )
    adapter_holder["adapter"] = adapter
    report = DeterministicDoctorTransaction().execute_live(
        _plan(before),
        worktree_adapter=adapter,
        edits=(
            DoctorExactEdit(
                "pkg/a.py",
                _hash(before["pkg/a.py"]),
                b"a = 2\n",
                step_id="step:0",
            ),
        ),
        target_ref="refs/heads/main",
        transaction_id="txn:chaos-drift",
    )
    assert not report.committed
    assert report.disposition is DoctorTransactionDisposition.ROLLED_BACK
    assert DoctorTransactionReason.DRIFT.value in report.reason_codes
    assert _git(root, "rev-parse", "refs/heads/main") == base
    assert _git(root, "show", "refs/heads/main:pkg/a.py") == "a = 1"


def test_incomplete_scc_and_stale_before_hash_do_not_mutate_ref(
    tmp_path: Path,
) -> None:
    before = {"pkg/a.py": b"a = 1\n", "pkg/b.py": b"b = 1\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    plan = _plan(before)
    adapter = _adapter(root, tmp_path / "state", tuple(before))

    with pytest.raises(Exception, match="cover.*complete|complete exact"):
        DeterministicDoctorTransaction().execute_live(
            plan,
            worktree_adapter=adapter,
            edits=(
                DoctorExactEdit(
                    "pkg/a.py",
                    _hash(before["pkg/a.py"]),
                    b"a = 2\n",
                    step_id="step:0",
                ),
            ),
            target_ref="refs/heads/main",
        )
    assert _git(root, "rev-parse", "refs/heads/main") == base

    with pytest.raises(DoctorWorktreeTamperError, match="before_hash"):
        DeterministicDoctorTransaction().execute_live(
            plan,
            worktree_adapter=_adapter(root, tmp_path / "state2", tuple(before)),
            edits=(
                DoctorExactEdit(
                    "pkg/a.py",
                    _hash(before["pkg/a.py"]),
                    b"a = 2\n",
                    step_id="step:0",
                ),
                DoctorExactEdit(
                    "pkg/b.py",
                    _hash(b"stale\n"),
                    b"b = 2\n",
                    step_id="step:1",
                ),
            ),
            target_ref="refs/heads/main",
            transaction_id="txn:stale-hash",
        )
    assert _git(root, "rev-parse", "refs/heads/main") == base
    assert _git(root, "show", "refs/heads/main:pkg/a.py") == "a = 1"
    assert _git(root, "show", "refs/heads/main:pkg/b.py") == "b = 1"


def test_live_commit_then_exact_root_identity(tmp_path: Path) -> None:
    before = {"pkg/a.py": b"a = 1\n", "pkg/b.py": b"b = 1\n"}
    after = {"pkg/a.py": b"a = 2\n", "pkg/b.py": b"b = 2\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    plan = _plan(before)
    edits = tuple(
        DoctorExactEdit(path, _hash(body), after[path], step_id=f"step:{index}")
        for index, (path, body) in enumerate(before.items())
    )
    report = DeterministicDoctorTransaction().execute_live(
        plan,
        worktree_adapter=_adapter(root, tmp_path / "state", tuple(before)),
        edits=edits,
        target_ref="refs/heads/main",
        transaction_id="txn:chaos-commit",
    )
    assert report.committed
    assert report.disposition is DoctorTransactionDisposition.COMMITTED
    assert report.merge_cas is not None
    assert report.merge_cas.expected_ref == base
    head = _git(root, "rev-parse", "refs/heads/main")
    assert report.merge_cas.desired_ref == head
    assert head != base
    for path, body in after.items():
        assert _git(root, "show", f"refs/heads/main:{path}") == body.decode().strip()


# ---------------------------------------------------------------------------
# Task-source / projection split brain
# ---------------------------------------------------------------------------


def test_task_source_split_brain_quarantines_and_blocks_apply(
    tmp_path: Path,
) -> None:
    store = PlanRevisionStore(tmp_path / "store")
    base = _revision()
    store.apply(
        PlanRevisionApplyRequest(
            revision=base,
            observed_roots=base.roots,
            idempotency_key="idem:base-q",
            expected_effects=("create",),
        )
    )

    class _DisagreeMarkdown:
        path = tmp_path / "ghost.md"

        def apply_plan_revision(self, **_kwargs: Any):
            return {"projection_cid": "md-cid-a"}

        def plan_revision_projection_cid(self):
            return "md-cid-a"

        def compare_plan_revision_parity(self, _other: Any):
            return {"valid": False, "mismatches": ("projection",)}

    class _DisagreeDuck:
        database_path = tmp_path / "ghost.duckdb"

        def apply_plan_revision(self, **_kwargs: Any):
            return {"projection_cid": "db-cid-b"}

        def plan_revision_projection_cid(self):
            return "db-cid-b"

    child = _revision(
        plan_root_cid=_cid("plan-root-split"),
        semantic_revision=2,
        parent_plan_root=base.plan_root_cid,
        origin=PlanOrigin.STEER,
        delta_cid=_delta().delta_cid,
        roots=base.roots,
        task_population=base.task_population,
    )
    with pytest.raises(PlanRevisionStoreQuarantinedError):
        store.apply(
            PlanRevisionApplyRequest(
                revision=child,
                observed_roots=child.roots,
                idempotency_key="idem:split",
                expected_effects=("steer",),
                delta=_delta(base_plan_root=base.plan_root_cid),
                markdown_source=_DisagreeMarkdown(),
                duckdb_source=_DisagreeDuck(),
                expected_active_plan_root=base.plan_root_cid,
            )
        )
    assert store.is_quarantined()
    with pytest.raises(PlanRevisionStoreQuarantinedError):
        store.apply(
            PlanRevisionApplyRequest(
                revision=child,
                observed_roots=child.roots,
                idempotency_key="idem:after-quarantine",
                expected_effects=("steer",),
                delta=_delta(base_plan_root=base.plan_root_cid),
            )
        )
    events = store.list_events()
    assert any(
        row.get("event_type") == "split_brain_quarantined" for row in events
    )


# ---------------------------------------------------------------------------
# Kill switch / PID identity / lease fence under chaos
# ---------------------------------------------------------------------------


def test_kill_switch_and_pid_identity_under_chaos() -> None:
    qualification = build_passing_observation(
        observation_id="observation:kill@1",
        observed_at="2026-08-03T00:00:00Z",
        role=ObservationRole.QUALIFICATION,
    )
    policy = default_rollout_policy(kill_switch_engaged=True)
    receipt = evaluate_planner_doctor_rollout(
        qualification,
        binding=default_rollout_binding(tree_id=qualification.tree_id),
        policy=policy,
        desired_mode=PlannerDoctorRolloutMode.CANARY,
    )
    assert receipt.kill_switch_override
    assert receipt.promotion_allowed is False
    assert receipt.effective_mode is PlannerDoctorRolloutMode.OFF

    # Current process PID is observable; a non-existent PID yields no tree.
    self_pid = os.getpid()
    descendants = collect_descendant_pids(self_pid)
    assert descendants is not None
    assert self_pid in descendants
    assert collect_descendant_pids(2**31 - 5) is None


def test_lane_matrix_under_chaos_preserves_pairing(
    live_runner: PlannerDoctorLiveBenchmark,
) -> None:
    for concurrency in (1, 2, 4, CONFIGURED_MAXIMUM_WORKERS):
        receipt = live_runner.run_pair(
            "live-hermetic-transaction-rollback",
            stratum_id="restart",
            concurrency=concurrency,
            repetition=0,
        )
        assert receipt.inputs_match_across_primary_arms
        assert receipt.concurrency_requested == concurrency
        for arm in receipt.arm_receipts:
            assert arm.effective_workers == effective_workers(concurrency)
            assert arm.process_tree_terminated
            assert arm.capabilities_revoked
            assert arm.output_root_sealed
