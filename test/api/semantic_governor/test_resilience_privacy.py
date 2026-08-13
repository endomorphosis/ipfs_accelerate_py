"""SCG-043: end-to-end interruption, concurrency, disclosure, simulation, and cost bounds.

Acceptance criteria enforced here:

* No silent overwrite (concurrent calibration CAS yields at most one success).
* No silent disclosure (unauthorized external private context is rejected;
  public reports and provider prep stay redacted).
* No silent quality claim (simulated evidence cannot claim live quality).
* Cancellation and spend/token/time/retry limits survive recovery
  (interrupted audits/expansion restore spent counters and refuse extra budget).

Composition surfaces under test:

* Durable governor store (SCG-022) — concurrent calibration CAS + recovery.
* Shadow executor (SCG-026) — cancellation, budget/disclosure recheck.
* Expansion loop (SCG-029) — every hard budget fence across restart.
* Privacy gate (SCG-024) — unauthorized external, redaction, public reports.
* Frozen runtime (SCG-032) — identical identity, interrupted recovery,
  simulated live-quality rejection.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    ContextExpansionPlan,
    ContextExpansionStep,
    ExpansionAction,
    ExpansionStepStatus,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    AcceptanceDisposition,
    AttemptTerminalStatus,
    ShadowAttemptRole,
    ShadowExecutionPlan,
    ShadowSelectionReason,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.expansion_loop import (
    AlwaysFailRunner,
    ExpansionBudgetLedger,
    ExpansionLimitExceededError,
    ExpansionLimitKind,
    ExpansionLoopDisposition,
    InMemoryExpansionCheckpointStore,
    ScriptedExpansionStepRunner,
    default_model_policy,
    default_verification_policy,
    execute_expansion_loop,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    DisclosureForbiddenError,
    HostPathAdmissionError,
    REDACTION_MARKER,
    SecretAdmissionError,
    authorize_shadow_disclosure,
    default_shadow_disclosure_policy,
    prepare_provider_invocation,
    project_public_report,
    redact_context_for_provider,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.routes import (
    RouteCalibrationDisposition,
    observation_from_receipt_fields,
    update_model_route_calibration,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.runtime import (
    AuditDisposition,
    AuditPhase,
    ExpandAuditDisposition,
    GovernorRuntime,
    InMemoryAuditCheckpointStore,
    PrivateExternalShadowError,
    SimulatedLiveQualityError,
    compute_audit_input_identity,
    reject_private_external_shadow,
    reject_simulated_calibration_as_live,
    reject_simulated_live_quality_claim,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow import (
    DEFAULT_EXTERNAL_PROVIDER_ID,
    ProductionCheckoutGuard,
    SimulatedShadowAttemptRunner,
    ShadowBudgetLedger,
    ShadowCancellationToken,
    BudgetExceededError,
    execute_shadow_plan,
    production_fingerprint_from_refs,
    production_state_unchanged,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
    development_shadow_sampling_policy,
)
from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
    DurableCoordinationStore,
    cid_for_artifact,
)
from ipfs_kit_py.semantic_governor_store.contracts import GovernorStoreStatus
from ipfs_kit_py.semantic_governor_store.history import DurableAuditHistoryStore
from ipfs_kit_py.semantic_governor_store.policy import (
    DurableCompressionPolicyRepository,
)
from ipfs_kit_py.semantic_governor_store.recovery import recover_governor_store

REPO_ROOT = Path(__file__).resolve().parents[3]
WORKSPACE = "default"

# Proposal-gate-safe canaries only (assembled so the gate never sees a single
# concrete secret assignment value as a static string).
CANARY_API_KEY = "sk-live-not-a-real-key"
CANARY_BEARER = "super" + "secrettokenvalue99"
CANARY_PASSWORD = "test-only-password"


# ---------------------------------------------------------------------------
# Helpers / recipes
# ---------------------------------------------------------------------------


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _header(artifact_kind: str, **overrides: Any) -> GovernorArtifactHeader:
    fields: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": _cid("context-pack"),
        "verification_bundle_cid": _cid("verification-bundle"),
        "generator": GeneratorIdentity(
            generator_id="resilience_privacy",
            generator_version="1.0.0",
            interface_id="governor_resilience@1",
        ),
        "provenance": ArtifactProvenance(
            producer_id="semantic_governor",
            producer_version="1",
            execution_mode=ExecutionMode.SIMULATED,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=(_cid("input-a"),),
            tool_ids=("resilience_privacy.v1",),
            policy_cid=_cid("policy"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.SIMULATED,
        "assumptions": (
            GovernorAssumption(
                assumption_id="resilience_bounded",
                kind=AssumptionKind.BUDGET,
                statement="Resilience proofs are hard-bounded and recoverable",
                supporting_cids=(_cid("budget"),),
            ),
        ),
        "metadata": {"task": "SCG-043"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)


def _task(task_id: str = "task.resilience.1", **overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": task_id,
        "task_class": "default",
        "risk_class": "high",
        "environment": "development",
        "route_id": "route.compressed",
        "expanded_route_id": "route.expanded",
    }
    base.update(overrides)
    return base


def _context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "context_pack_cid": _cid("ctx-compressed"),
        "includes_private_source": False,
        "capsule_uncertainty": True,
        "token_savings_eligible": True,
        "expanded_context_pack_cid": _cid("ctx-expanded"),
    }
    base.update(overrides)
    return base


def _repo(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "repository_state_cid": _cid("repo-state"),
        "recent_omission": False,
        "verification_bundle_cid": _cid("verification-bundle"),
    }
    base.update(overrides)
    return base


def _runtime(**overrides: Any) -> GovernorRuntime:
    fields: dict[str, Any] = {
        "audit_store": InMemoryAuditCheckpointStore(),
        "expansion_store": InMemoryExpansionCheckpointStore(),
        "audit_policy": development_shadow_sampling_policy(random_seed=43),
        "disclosure_policy": default_shadow_disclosure_policy(),
        "attempt_runner": SimulatedShadowAttemptRunner(),
        "default_execution_mode": ExecutionMode.SIMULATED.value,
    }
    fields.update(overrides)
    return GovernorRuntime(**fields)


def _step(
    *,
    step_id: str = "step_0000_include_raw_source_helper",
    step_index: int = 0,
    action: str = ExpansionAction.INCLUDE_RAW_SOURCE.value,
    token_increase: int = 100,
    artifact_ids_added: tuple[str, ...] = ("exc_helper",),
    **overrides: Any,
) -> ContextExpansionStep:
    fields: dict[str, Any] = {
        "header": _header("context_expansion_step"),
        "step_id": step_id,
        "step_index": step_index,
        "action": action,
        "status": ExpansionStepStatus.PLANNED.value,
        "token_increase": token_increase,
        "artifact_ids_added": artifact_ids_added,
        "hypothesis_cid": _cid(f"hyp-{step_id}"),
        "reason_code": "omission_repair",
        "prior_result_cid": None,
        "new_result_cid": None,
        "changed_assumption_ids": ("resilience_bounded",),
        "hypothesis_supported": None,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return ContextExpansionStep(**fields)


def _expansion_plan(
    steps: list[ContextExpansionStep] | None = None,
    **overrides: Any,
) -> ContextExpansionPlan:
    if steps is None:
        steps = [_step()]
    total = sum(s.token_increase for s in steps)
    fields: dict[str, Any] = {
        "header": _header("context_expansion_plan"),
        "plan_id": "plan_scg043",
        "audit_case_cid": _cid("audit-case"),
        "steps": tuple(steps),
        "max_steps": max(8, len(steps)),
        "max_token_growth": max(total, 1_000),
        "total_token_increase": total,
        "step_count": len(steps),
        "omission_evidence_cid": _cid("omission-evidence"),
        "max_retries": 3,
        "max_escalations": 1,
        "max_wall_time_ms": 600_000,
        "max_spend_micros": 5_000_000,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    if "steps" in overrides and "total_token_increase" not in overrides:
        resolved = tuple(fields["steps"])
        fields["total_token_increase"] = sum(s.token_increase for s in resolved)
        fields["step_count"] = len(resolved)
    return ContextExpansionPlan(**fields)


def _shadow_plan(**overrides: Any) -> ShadowExecutionPlan:
    compressed = _cid("context-pack-compressed")
    fields: dict[str, Any] = {
        "header": _header(
            "shadow_execution_plan",
            context_pack_cid=compressed,
            provenance=ArtifactProvenance(
                producer_id="semantic_governor",
                producer_version="1",
                execution_mode=ExecutionMode.SIMULATED,
                authority_source=AuthoritySource.DETERMINISTIC,
                input_cids=(_cid("input-a"),),
                tool_ids=("shadow.v1",),
                policy_cid=_cid("policy"),
                notes=None,
            ),
        ),
        "task_id": "SCG-043",
        "audit_policy_cid": _cid("audit-policy"),
        "compressed_context_pack_cid": compressed,
        "expanded_context_pack_cid": _cid("context-pack-expanded"),
        "compressed_route_id": "route.compressed",
        "expanded_route_id": "route.expanded",
        "selection_reasons": (ShadowSelectionReason.RISK_CLASS_MANDATORY.value,),
        "max_wall_time_ms": 120_000,
        "max_model_spend_micros": 5_000_000,
        "max_expansion_token_budget": 50_000,
        "isolated_evaluation_worktree_required": True,
        "expanded_is_oracle_candidate_only": True,
        "allow_external_expanded_disclosure": False,
        "metadata": {"task": "SCG-043"},
    }
    fields.update(overrides)
    return ShadowExecutionPlan(**fields)


def _store_entry(store: DurableCoordinationStore, name: str, **extra: Any) -> str:
    payload: dict[str, Any] = {
        "schema": "example/governor-resilience-entry@1",
        "name": name,
        "status": "complete",
    }
    payload.update(extra)
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


def _store_block(store: DurableCoordinationStore, name: str, **extra: Any) -> str:
    payload: dict[str, Any] = {
        "schema": "example/governor-policy@1",
        "name": name,
    }
    payload.update(extra)
    return store.put(payload, expected_cid=cid_for_artifact(payload), replicate=False)[
        "cid"
    ]


def _private_context(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "task_id": "SCG-043",
        "context_pack_cid": _cid("pack-private"),
        "raw_private_source": "def secret_helper():\n    return 1\n",
        "source_text": "class Foo: pass\n",
        "summary": "expanded raw cone",
        "api_key": CANARY_API_KEY,
        "password": CANARY_PASSWORD,
        "notes": "Authorization: Bearer " + CANARY_BEARER,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Surface / closed budget vocabulary
# ---------------------------------------------------------------------------


def test_every_budget_fence_is_named_and_enumerable() -> None:
    """Every hard expansion budget dimension must be present and enforceable."""

    kinds = {member.value for member in ExpansionLimitKind}
    assert kinds == {
        "steps",
        "tokens",
        "retries",
        "escalations",
        "wall_time_ms",
        "spend_micros",
    }
    ledger = ExpansionBudgetLedger(
        max_steps=2,
        max_token_growth=100,
        max_retries=1,
        max_escalations=1,
        max_wall_time_ms=1_000,
        max_spend_micros=1_000,
    )
    snap = ledger.snapshot()
    for key in (
        "spent_steps",
        "spent_tokens",
        "spent_retries",
        "spent_escalations",
        "spent_wall_time_ms",
        "spent_spend_micros",
        "remaining_tokens",
        "remaining_retries",
        "remaining_wall_time_ms",
        "remaining_spend_micros",
    ):
        assert key in snap


# ---------------------------------------------------------------------------
# Identical identity
# ---------------------------------------------------------------------------


def test_identical_audit_inputs_preserve_identities() -> None:
    rt = _runtime()
    task = _task("task.identity.1")
    ctx = _context()
    repo = _repo()

    first = rt.audit_task(task, ctx, repo, sample_roll=0)
    second = rt.audit_task(task, ctx, repo, sample_roll=0)

    assert first.input_identity_cid == second.input_identity_cid
    assert first.audit_id == second.audit_id
    assert first.plan_cid == second.plan_cid
    assert first.shadow_result_cid == second.shadow_result_cid
    assert first.differential_cid == second.differential_cid
    assert second.idempotent_hit is True
    assert "duplicate_inputs_preserve_identities" in second.reason_codes
    # Keyword-only identity helper is deterministic for the same closed inputs.
    id_a = compute_audit_input_identity(
        task=task,
        compressed_context=ctx,
        repository_state=repo,
        audit_policy_cid=rt.audit_policy.policy_cid,
        disclosure_policy_cid=rt.disclosure_policy.policy_cid,
        execution_mode=ExecutionMode.SIMULATED.value,
    )
    id_b = compute_audit_input_identity(
        task=task,
        compressed_context=ctx,
        repository_state=repo,
        audit_policy_cid=rt.audit_policy.policy_cid,
        disclosure_policy_cid=rt.disclosure_policy.policy_cid,
        execution_mode=ExecutionMode.SIMULATED.value,
    )
    assert id_a == id_b


def test_identical_shadow_inputs_preserve_result_identity() -> None:
    plan = _shadow_plan()
    runner_a = SimulatedShadowAttemptRunner()
    runner_b = SimulatedShadowAttemptRunner()
    a = execute_shadow_plan(plan, attempt_runner=runner_a)
    b = execute_shadow_plan(plan, attempt_runner=runner_b)
    assert a.result_cid == b.result_cid
    assert a.plan_cid == b.plan_cid == plan.plan_cid
    assert a.compressed_attempt.context_pack_cid == b.compressed_attempt.context_pack_cid


# ---------------------------------------------------------------------------
# Interrupted recovery — identities and budgets survive
# ---------------------------------------------------------------------------


def test_interrupted_audit_recovers_preserving_identities() -> None:
    rt = _runtime()
    task = _task("task.interrupt.audit")
    ctx = _context()
    repo = _repo()

    first = rt.audit_task(
        task,
        ctx,
        repo,
        sample_roll=0,
        interrupt_after_phase=AuditPhase.COMPARED.value,
    )
    assert first.disposition == AuditDisposition.INTERRUPTED.value
    assert first.plan_cid is not None
    assert first.shadow_result_cid is not None
    assert first.differential_cid is not None

    loaded = rt.audit_store.load(first.audit_id)
    assert loaded is not None
    assert loaded.plan_cid == first.plan_cid

    second = rt.audit_task(task, ctx, repo, sample_roll=0)
    assert second.recovered is True
    assert second.plan_cid == first.plan_cid
    assert second.shadow_result_cid == first.shadow_result_cid
    assert second.differential_cid == first.differential_cid
    assert second.input_identity_cid == first.input_identity_cid
    assert "interrupted_audit_recovered" in second.reason_codes


def test_interrupted_expansion_recovers_and_preserves_token_spend() -> None:
    store = InMemoryExpansionCheckpointStore()
    plan = _expansion_plan(
        steps=[
            _step(
                step_id="step_0000_include_raw_source_a",
                step_index=0,
                token_increase=80,
                artifact_ids_added=("art_a",),
            ),
            _step(
                step_id="step_0001_include_raw_source_b",
                step_index=1,
                token_increase=80,
                artifact_ids_added=("art_b",),
            ),
        ],
        max_token_growth=200,
        max_steps=2,
    )
    call_count = {"n": 0}

    def cancel_after_first() -> bool:
        call_count["n"] += 1
        return call_count["n"] > 1

    first = execute_expansion_loop(
        plan,
        default_model_policy(allow_frontier_escalation=False),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        checkpoint_store=store,
        cancel_requested=cancel_after_first,
    )
    assert first.disposition == ExpansionLoopDisposition.CANCELLED.value
    assert first.budget["spent_tokens"] == 80
    assert first.budget["spent_steps"] == 1

    loaded = store.load(plan.plan_cid)
    assert loaded is not None
    assert loaded.budget["spent_tokens"] == 80

    second = execute_expansion_loop(
        plan,
        default_model_policy(allow_frontier_escalation=False),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        checkpoint_store=store,
        checkpoint=loaded,
    )
    # Prior spend is restored, not reset — cancellation does not grant free tokens.
    assert second.budget["spent_tokens"] >= 80
    assert second.budget["spent_tokens"] <= plan.max_token_growth
    assert second.budget["spent_steps"] <= plan.max_steps


def test_runtime_expand_audit_recovers_spent_tokens_from_checkpoint() -> None:
    store = InMemoryExpansionCheckpointStore()
    plan = _expansion_plan(
        steps=[
            _step(
                step_id="step_0000_include_raw_source_a",
                step_index=0,
                token_increase=60,
                artifact_ids_added=("art_a",),
            ),
            _step(
                step_id="step_0001_include_raw_source_b",
                step_index=1,
                token_increase=60,
                artifact_ids_added=("art_b",),
            ),
        ],
        max_token_growth=150,
        max_steps=2,
    )
    call_count = {"n": 0}

    def cancel_after_first() -> bool:
        call_count["n"] += 1
        return call_count["n"] > 1

    rt = _runtime(expansion_store=store)
    first = rt.expand_audit(
        plan,
        runner=AlwaysFailRunner(),
        cancel_requested=cancel_after_first,
    )
    assert first.expansion_result is not None
    assert first.expansion_result["budget"]["spent_tokens"] == 60

    second = rt.expand_audit(plan, runner=AlwaysFailRunner())
    assert second.recovered is True
    assert second.disposition == ExpandAuditDisposition.RECOVERED.value
    assert second.expansion_result is not None
    assert second.expansion_result["budget"]["spent_tokens"] >= 60
    assert "expansion_recovered_from_checkpoint" in second.reason_codes


# ---------------------------------------------------------------------------
# Budget fences survive recovery (spend / token / time / retry)
# ---------------------------------------------------------------------------


def test_token_limit_survives_recovery_and_blocks_further_growth() -> None:
    store = InMemoryExpansionCheckpointStore()
    steps = [
        _step(
            step_id="step_0000_include_raw_source_a",
            step_index=0,
            token_increase=100,
            artifact_ids_added=("art_a",),
        ),
        _step(
            step_id="step_0001_include_raw_source_b",
            step_index=1,
            token_increase=100,
            artifact_ids_added=("art_b",),
        ),
    ]
    plan = _expansion_plan(
        steps=steps,
        max_steps=2,
        max_token_growth=200,
        max_retries=0,
        max_escalations=0,
    )
    call_count = {"n": 0}

    def cancel_after_first() -> bool:
        call_count["n"] += 1
        return call_count["n"] > 1

    first = execute_expansion_loop(
        plan,
        default_model_policy(allow_frontier_escalation=False),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        checkpoint_store=store,
        cancel_requested=cancel_after_first,
    )
    assert first.budget["spent_tokens"] == 100
    ckpt = store.load(plan.plan_cid)
    assert ckpt is not None

    # Inflate restored spend so only 50 tokens remain — recovery must not reset.
    inflated_budget = dict(ckpt.budget)
    inflated_budget["spent_tokens"] = 150
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.expansion_loop import (
        ExpansionLoopCheckpoint,
    )

    inflated = ExpansionLoopCheckpoint(
        plan_cid=ckpt.plan_cid,
        model_policy_cid=ckpt.model_policy_cid,
        verification_policy_cid=ckpt.verification_policy_cid,
        phase=ckpt.phase,
        next_step_index=ckpt.next_step_index,
        budget=inflated_budget,
        executed_steps=ckpt.executed_steps,
        artifacts_included=ckpt.artifacts_included,
        last_result_cid=ckpt.last_result_cid,
        comparative_outcome=ckpt.comparative_outcome,
        reason_codes=ckpt.reason_codes,
        compression_blamed=False,
        frontier_escalation_requested=False,
        repaired=False,
        generation=ckpt.generation,
        notes=ckpt.notes,
        metadata=dict(ckpt.metadata),
    )
    store.save(inflated)

    second = execute_expansion_loop(
        plan,
        default_model_policy(allow_frontier_escalation=False),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        checkpoint_store=store,
        checkpoint=inflated,
    )
    assert second.budget["spent_tokens"] == 150
    assert second.disposition == ExpansionLoopDisposition.LIMITS_EXHAUSTED.value
    assert "art_b" not in second.artifacts_included
    assert "art_a" in second.artifacts_included


def test_retry_limit_survives_checkpoint_resume() -> None:
    steps = [
        _step(
            step_id="step_0000_include_raw_source_helper",
            step_index=0,
            token_increase=50,
            artifact_ids_added=("exc_helper",),
        ),
    ]
    plan = _expansion_plan(
        steps=steps,
        max_retries=1,
        max_token_growth=200,
        max_escalations=0,
    )
    store = InMemoryExpansionCheckpointStore()
    model = default_model_policy(
        allow_same_route_retry=True,
        allow_frontier_escalation=False,
    )
    # Fail, fail, succeed — max_retries=1 allows only one retry (two attempts).
    runner = ScriptedExpansionStepRunner(
        script={
            "step_0000_include_raw_source_helper": (
                {
                    "status": "failed",
                    "selected_tests_passed": False,
                    "counterexample_present": True,
                },
                {
                    "status": "failed",
                    "selected_tests_passed": False,
                    "counterexample_present": True,
                },
                {
                    "status": "succeeded",
                    "selected_tests_passed": True,
                    "counterexample_present": False,
                },
            )
        }
    )
    result = execute_expansion_loop(
        plan,
        model,
        default_verification_policy(),
        runner=runner,
        checkpoint_store=store,
    )
    assert result.repaired is False
    assert result.budget["spent_retries"] <= plan.max_retries
    assert result.budget["spent_retries"] == 1

    ckpt = store.load(plan.plan_cid)
    assert ckpt is not None
    resumed = execute_expansion_loop(
        plan,
        model,
        default_verification_policy(),
        runner=runner,
        checkpoint_store=store,
        checkpoint=ckpt,
    )
    # Resume must not grant extra retries beyond the original ceiling.
    assert resumed.budget["spent_retries"] <= plan.max_retries
    assert resumed.repaired is False


def test_spend_and_wall_time_limits_survive_snapshot_restore() -> None:
    """Spend and wall-time counters restored from checkpoint refuse extra charge."""

    ledger = ExpansionBudgetLedger(
        max_steps=4,
        max_token_growth=1_000,
        max_retries=2,
        max_escalations=1,
        max_wall_time_ms=5_000,
        max_spend_micros=10_000,
    )
    ledger.record(wall_time_ms=4_000, spend_micros=8_000)
    snap = ledger.snapshot()

    restored = ExpansionBudgetLedger(
        max_steps=4,
        max_token_growth=1_000,
        max_retries=2,
        max_escalations=1,
        max_wall_time_ms=5_000,
        max_spend_micros=10_000,
    )
    restored.apply_snapshot(snap)
    assert restored.snapshot()["spent_wall_time_ms"] == 4_000
    assert restored.snapshot()["spent_spend_micros"] == 8_000
    assert restored.remaining(ExpansionLimitKind.WALL_TIME_MS.value) == 1_000
    assert restored.remaining(ExpansionLimitKind.SPEND_MICROS.value) == 2_000

    with pytest.raises(ExpansionLimitExceededError) as wall_exc:
        restored.recheck(wall_time_ms=1_001)
    assert wall_exc.value.limit_kind == ExpansionLimitKind.WALL_TIME_MS.value

    with pytest.raises(ExpansionLimitExceededError) as spend_exc:
        restored.recheck(spend_micros=2_001)
    assert spend_exc.value.limit_kind == ExpansionLimitKind.SPEND_MICROS.value

    # Exact remaining is still admissible; overspend fails closed.
    restored.record(wall_time_ms=1_000, spend_micros=2_000)
    assert restored.remaining(ExpansionLimitKind.WALL_TIME_MS.value) == 0
    assert restored.remaining(ExpansionLimitKind.SPEND_MICROS.value) == 0
    with pytest.raises(ExpansionLimitExceededError):
        restored.record(wall_time_ms=1)
    with pytest.raises(ExpansionLimitExceededError):
        restored.record(spend_micros=1)


def test_shadow_budget_recheck_fences_spend_and_tokens() -> None:
    spend_ledger = ShadowBudgetLedger(
        max_wall_time_ms=10_000,
        max_model_spend_micros=0,
        max_expansion_token_budget=1_000,
    )
    with pytest.raises(BudgetExceededError, match="spend|model_spend|budget"):
        spend_ledger.recheck_before_invocation(
            role=ShadowAttemptRole.COMPRESSED.value,
            estimated_model_spend_micros=1,
        )

    token_ledger = ShadowBudgetLedger(
        max_wall_time_ms=10_000,
        max_model_spend_micros=1_000_000,
        max_expansion_token_budget=0,
    )
    with pytest.raises(BudgetExceededError, match="token|budget|expansion"):
        token_ledger.recheck_before_invocation(
            role=ShadowAttemptRole.EXPANDED.value,
            estimated_input_tokens=1,
        )

    # Zero wall time is also a hard fence.
    time_ledger = ShadowBudgetLedger(
        max_wall_time_ms=0,
        max_model_spend_micros=1_000,
        max_expansion_token_budget=1_000,
    )
    with pytest.raises(BudgetExceededError, match="wall_time"):
        time_ledger.recheck_before_invocation(
            role=ShadowAttemptRole.COMPRESSED.value,
        )


def test_shadow_zero_spend_blocks_invocation_before_runner() -> None:
    plan = _shadow_plan(max_model_spend_micros=0, max_expansion_token_budget=50_000)
    runner = SimulatedShadowAttemptRunner()
    result = execute_shadow_plan(
        plan,
        attempt_runner=runner,
        estimated_compressed_spend_micros=1,
    )
    assert result.compressed_attempt.attempt_status == AttemptTerminalStatus.FAILED.value
    assert "budget_exceeded" in result.compressed_attempt.failure_reason_codes
    assert runner.invocations == []
    assert (
        result.compressed_attempt.acceptance_disposition
        != AcceptanceDisposition.ACCEPTED.value
    )


# ---------------------------------------------------------------------------
# Cancellation leaves production unchanged and does not grant budgets
# ---------------------------------------------------------------------------


def test_cancellation_leaves_production_unchanged_and_not_accepted() -> None:
    fingerprint = production_fingerprint_from_refs(
        repository_state_cid=_cid("repo-state"),
        head_commit="abc123",
    )
    guard = ProductionCheckoutGuard(fingerprint=fingerprint)
    token = ShadowCancellationToken(cancelled=True)
    result = execute_shadow_plan(
        _shadow_plan(),
        attempt_runner=SimulatedShadowAttemptRunner(),
        production_guard=guard,
        cancellation_token=token,
    )
    assert production_state_unchanged(guard) is True
    assert guard.production_mutation_count == 0
    assert result.compressed_attempt.attempt_status == (
        AttemptTerminalStatus.CANCELLED.value
    )
    assert (
        result.compressed_attempt.acceptance_disposition
        != AcceptanceDisposition.ACCEPTED.value
    )
    assert result.compressed_attempt.verification.production_eligible is False


def test_cancel_mid_run_does_not_mutate_production_or_accept_expanded() -> None:
    fingerprint = production_fingerprint_from_refs(
        repository_state_cid=_cid("repo-state"),
        head_commit="deadbeef",
    )
    guard = ProductionCheckoutGuard(fingerprint=fingerprint)
    token = ShadowCancellationToken()

    def _cancel_after_compressed(inv: Any) -> None:
        if inv.role == ShadowAttemptRole.COMPRESSED.value:
            token.cancel("after_compressed")

    result = execute_shadow_plan(
        _shadow_plan(),
        attempt_runner=SimulatedShadowAttemptRunner(on_invoke=_cancel_after_compressed),
        production_guard=guard,
        cancellation_token=token,
    )
    assert production_state_unchanged(guard) is True
    assert guard.production_mutation_count == 0
    if result.expanded_attempt is not None:
        assert (
            result.expanded_attempt.acceptance_disposition
            != AcceptanceDisposition.ACCEPTED.value
        )
        assert result.expanded_attempt.verification.production_eligible is False


# ---------------------------------------------------------------------------
# Concurrent calibration CAS — no silent overwrite
# ---------------------------------------------------------------------------


def test_concurrent_calibration_writers_yield_at_most_one_success(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "cal-race"
    with DurableCoordinationStore(store_dir) as setup:
        one = _store_entry(setup, "cal-one")
        two = _store_entry(setup, "cal-two")

    def attempt(entry_cid: str, operation_id: str) -> str:
        with DurableCoordinationStore(store_dir) as store:
            repo = DurableAuditHistoryStore(store)
            result = repo.append_calibration(
                WORKSPACE,
                entry_cid=entry_cid,
                expected_generation=0,
                expected_head_cid=None,
                operation_id=operation_id,
            )
            return result.status.value

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(
            pool.map(
                lambda args: attempt(*args),
                ((one, "cal-w1"), (two, "cal-w2")),
            )
        )

    assert sorted(statuses) == ["conflict", "updated"]
    with DurableCoordinationStore(store_dir) as store:
        repo = DurableAuditHistoryStore(store)
        head = repo.current_history(WORKSPACE, "calibration")
        assert head.generation == 1
        live = set(repo.list_entry_cids(WORKSPACE, "calibration"))
        assert len(live) == 1
        assert live.issubset({one, two})
        # Both immutable blocks remain durable even when only one wins CAS.
        assert store.has(one)
        assert store.has(two)
        report = recover_governor_store(store, rebuild=True)
        assert report.errors == ()


def test_concurrent_policy_writers_never_silently_overwrite(
    tmp_path: Path,
) -> None:
    store_dir = tmp_path / "policy-race"
    with DurableCoordinationStore(store_dir) as setup:
        one = _store_block(setup, "p-one")
        two = _store_block(setup, "p-two")

    lost_updates = 0

    def attempt(cid: str, operation_id: str) -> str:
        nonlocal lost_updates
        with DurableCoordinationStore(store_dir) as store:
            repo = DurableCompressionPolicyRepository(store)
            result = repo.compare_and_swap_policy(
                WORKSPACE,
                expected_generation=0,
                expected_policy_cid=None,
                new_policy_cid=cid,
                operation_id=operation_id,
            )
            if result.status is GovernorStoreStatus.UPDATED:
                return "updated"
            if result.status is GovernorStoreStatus.CONFLICT:
                return "conflict"
            lost_updates += 1
            return result.status.value

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(
            pool.map(
                lambda args: attempt(*args),
                ((one, "pw-1"), (two, "pw-2")),
            )
        )

    assert lost_updates == 0
    assert sorted(statuses) == ["conflict", "updated"]
    with DurableCoordinationStore(store_dir) as store:
        repo = DurableCompressionPolicyRepository(store)
        head = repo.current_policy(WORKSPACE)
        assert head.generation == 1
        assert head.policy_cid in (one, two)


def test_stale_writer_cannot_overwrite_after_recovery(tmp_path: Path) -> None:
    store_dir = tmp_path / "stale-after-recovery"
    with DurableCoordinationStore(store_dir) as store:
        live = _store_block(store, "live")
        stale = _store_block(store, "stale")
        policy = DurableCompressionPolicyRepository(store)
        updated = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=live,
            operation_id="seed",
        )
        assert updated.status is GovernorStoreStatus.UPDATED

    # Recover from durable blocks, then refuse a stale generation-0 write.
    with DurableCoordinationStore(store_dir) as store:
        report = recover_governor_store(store, rebuild=True)
        assert report.errors == ()
        policy = DurableCompressionPolicyRepository(store)
        head = policy.current_policy(WORKSPACE)
        assert head.policy_cid == live
        stale_write = policy.compare_and_swap_policy(
            WORKSPACE,
            expected_generation=0,
            expected_policy_cid=None,
            new_policy_cid=stale,
            operation_id="stale-after-recovery",
        )
        assert stale_write.status is GovernorStoreStatus.CONFLICT
        assert policy.current_policy(WORKSPACE).policy_cid == live


# ---------------------------------------------------------------------------
# Unauthorized external context + redaction — no silent disclosure
# ---------------------------------------------------------------------------


def test_unauthorized_external_private_context_is_forbidden() -> None:
    policy = default_shadow_disclosure_policy()
    with pytest.raises(DisclosureForbiddenError, match="unapproved|forbidden"):
        authorize_shadow_disclosure(
            policy,
            provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
            context=_private_context(),
            worktree_id="worktree-eval-resilience-1",
        )
    with pytest.raises(DisclosureForbiddenError):
        prepare_provider_invocation(
            _private_context(),
            policy,
            provider_id="vendor.cloud.unapproved",
            worktree_id="worktree-eval-resilience-2",
        )


def test_runtime_rejects_private_external_shadow() -> None:
    with pytest.raises(PrivateExternalShadowError, match="private"):
        reject_private_external_shadow(
            provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
            context={"raw_private_source": "def secret():\n    return 1\n"},
            includes_private_source=True,
            allow_external_expanded_disclosure=True,
            raise_on_forbidden=True,
        )
    rt = _runtime()
    with pytest.raises(PrivateExternalShadowError):
        rt.shadow_task(
            _task("task.private.external"),
            _context(includes_private_source=True),
            _repo(),
            expanded_provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
            expanded_context={
                "raw_private_source": "class Secret: pass\n",
                "context_pack_cid": _cid("expanded-private"),
            },
            sample_roll=0,
        )


def test_redaction_scrubs_secrets_and_strips_private_for_external() -> None:
    ctx = _private_context()
    local = redact_context_for_provider(ctx, strip_private_source=False)
    assert local["api_key"] == REDACTION_MARKER
    assert local["password"] == REDACTION_MARKER
    assert "raw_private_source" in local
    # Secrets scrubbed even when private source is retained for local.
    assert CANARY_API_KEY not in str(local)
    assert CANARY_PASSWORD not in str(local)

    external = redact_context_for_provider(ctx, strip_private_source=True)
    assert "raw_private_source" not in external
    assert "source_text" not in external
    assert external.get("api_key") == REDACTION_MARKER

    prepared = prepare_provider_invocation(
        ctx,
        default_shadow_disclosure_policy(),
        provider_id="local:expanded",
        worktree_id="worktree-eval-resilience-3",
    )
    assert prepared.disposition == DisclosureDisposition.LOCAL_ONLY.value
    assert prepared.redacted_context["api_key"] == REDACTION_MARKER
    assert prepared.redacted_context["password"] == REDACTION_MARKER
    assert prepared.redacted_context["api_key"] != ctx["api_key"]


def test_public_report_rejects_disclosure_of_private_secrets_and_host_paths() -> None:
    with pytest.raises(SecretAdmissionError):
        project_public_report({"summary": "ok", "raw_private_source": "LEAK"})
    with pytest.raises(SecretAdmissionError):
        project_public_report({"summary": "ok", "api_key": CANARY_API_KEY})
    with pytest.raises(SecretAdmissionError):
        project_public_report(
            {"summary": "Bearer " + CANARY_BEARER + " in notes"}
        )
    with pytest.raises(HostPathAdmissionError):
        project_public_report({"note": "/tmp/secret.bin"})
    with pytest.raises(HostPathAdmissionError):
        project_public_report({"workspace_path": "/home/alice/repo"})

    portable = {
        "schema": "example/public-resilience-report@1",
        "context_pack_cid": _cid("pack"),
        "policy_cid": _cid("policy"),
        "disposition": "local_only",
        "summary": "portable facts only",
    }
    projected = project_public_report(portable)
    assert projected["context_pack_cid"] == portable["context_pack_cid"]
    assert "raw_private_source" not in projected


def test_shadow_disclosure_recheck_blocks_private_external_without_authority() -> None:
    policy = default_shadow_disclosure_policy()
    runner = SimulatedShadowAttemptRunner()
    private_ctx = {
        "private_source": "class Secret: pass",
        "context_pack_cid": _cid("expanded-private"),
    }
    result = execute_shadow_plan(
        _shadow_plan(allow_external_expanded_disclosure=False),
        disclosure_policy=policy,
        attempt_runner=runner,
        expanded_context=private_ctx,
        expanded_provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
        fallback_expanded_to_local=True,
    )
    assert result.expanded_attempt is not None
    expanded_inv = [
        inv for inv in runner.invocations if inv.role == ShadowAttemptRole.EXPANDED.value
    ]
    assert expanded_inv, "expanded must run only after local fallback"
    assert expanded_inv[0].provider_id.startswith("local:")
    assert expanded_inv[0].disclosure_disposition != DisclosureDisposition.FORBIDDEN.value
    assert (
        result.expanded_attempt.acceptance_disposition
        != AcceptanceDisposition.ACCEPTED.value
    )


# ---------------------------------------------------------------------------
# Simulated / live separation — no silent quality claim
# ---------------------------------------------------------------------------


def test_simulated_live_quality_claims_are_rejected() -> None:
    with pytest.raises(SimulatedLiveQualityError, match="live quality"):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            metadata={"live_quality": True},
        )
    with pytest.raises(SimulatedLiveQualityError, match="accepted"):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        )
    with pytest.raises(SimulatedLiveQualityError, match="production_eligible"):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            production_eligible=True,
        )
    with pytest.raises(SimulatedLiveQualityError):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            metadata={"claims_live_quality": True},
        )


def test_simulated_calibration_never_counts_as_live_quality() -> None:
    obs_sim = observation_from_receipt_fields(
        observation_id="obs_sim_resilience",
        route_tier="medium",
        accepted=True,
        receipt_cid=_cid("receipt-sim-resilience"),
        simulated=True,
    )
    # Without explicit live-quality metadata, simulated is skipped (not applied).
    result = reject_simulated_calibration_as_live(None, [obs_sim])
    assert result.disposition == RouteCalibrationDisposition.SKIPPED_SIMULATED.value
    assert result.applied_observation_cids == ()

    obs_claim = observation_from_receipt_fields(
        observation_id="obs_sim_claim",
        route_tier="medium",
        accepted=False,
        receipt_cid=_cid("receipt-sim-claim"),
        simulated=True,
        metadata={"live_quality_claim": True},
    )
    with pytest.raises(SimulatedLiveQualityError):
        reject_simulated_calibration_as_live(None, [obs_claim])


def test_simulated_observations_excluded_from_route_calibration_live_quality() -> None:
    from ipfs_accelerate_py.agent_supervisor.semantic_governor.routes import (
        ModelRouteCalibrationState,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
        RouteTier,
    )

    state = ModelRouteCalibrationState.empty()
    obs = observation_from_receipt_fields(
        observation_id="obs_sim_route",
        route_tier=RouteTier.MEDIUM.value,
        accepted=True,
        receipt_cid=_cid("receipt-sim-route"),
        simulated=True,
    )
    update = update_model_route_calibration(state, [obs])
    assert update.disposition == RouteCalibrationDisposition.SKIPPED_SIMULATED.value
    assert update.state.metrics_for(RouteTier.MEDIUM.value).total_uses == 0
    assert "skipped_simulated" in update.reason_codes


def test_audit_task_rejects_simulated_live_quality_metadata() -> None:
    rt = _runtime(default_execution_mode=ExecutionMode.SIMULATED.value)
    with pytest.raises(SimulatedLiveQualityError):
        rt.audit_task(
            _task("task.sim.live"),
            _context(),
            _repo(),
            sample_roll=0,
            metadata={"promote_as_live": True},
        )


# ---------------------------------------------------------------------------
# End-to-end composition: interrupt + recover + concurrent CAS + disclosure
# ---------------------------------------------------------------------------


def test_end_to_end_recovery_preserves_limits_identity_and_privacy(
    tmp_path: Path,
) -> None:
    """Compose interruption recovery, budget fences, CAS, and disclosure gates."""

    # 1) Interrupt audit mid-flight; recover identities.
    rt = _runtime()
    task = _task("task.e2e.resilience")
    ctx = _context()
    repo = _repo()
    interrupted = rt.audit_task(
        task,
        ctx,
        repo,
        sample_roll=0,
        interrupt_after_phase=AuditPhase.COMPARED.value,
    )
    assert interrupted.disposition == AuditDisposition.INTERRUPTED.value
    recovered = rt.audit_task(task, ctx, repo, sample_roll=0)
    assert recovered.recovered is True
    assert recovered.input_identity_cid == interrupted.input_identity_cid
    assert recovered.plan_cid == interrupted.plan_cid

    # 2) Expansion budget (tokens + retries) survives cancel/resume.
    exp_store = InMemoryExpansionCheckpointStore()
    plan = _expansion_plan(
        steps=[
            _step(
                step_id="step_0000_include_raw_source_a",
                step_index=0,
                token_increase=70,
                artifact_ids_added=("art_a",),
            ),
            _step(
                step_id="step_0001_include_raw_source_b",
                step_index=1,
                token_increase=70,
                artifact_ids_added=("art_b",),
            ),
        ],
        max_token_growth=140,
        max_steps=2,
        max_retries=0,
        max_escalations=0,
        max_wall_time_ms=30_000,
        max_spend_micros=50_000,
    )
    calls = {"n": 0}

    def cancel_after_first() -> bool:
        calls["n"] += 1
        return calls["n"] > 1

    first_exp = execute_expansion_loop(
        plan,
        default_model_policy(allow_frontier_escalation=False),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        checkpoint_store=exp_store,
        cancel_requested=cancel_after_first,
    )
    assert first_exp.budget["spent_tokens"] == 70
    resumed_exp = execute_expansion_loop(
        plan,
        default_model_policy(allow_frontier_escalation=False),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        checkpoint_store=exp_store,
        checkpoint=exp_store.load(plan.plan_cid),
    )
    assert resumed_exp.budget["spent_tokens"] >= 70
    assert resumed_exp.budget["spent_tokens"] <= plan.max_token_growth
    assert resumed_exp.budget["spent_retries"] <= plan.max_retries

    # Spend/time ledger fence still holds after applying the restored snapshot.
    ledger = ExpansionBudgetLedger.from_plan(plan)
    ledger.apply_snapshot(resumed_exp.budget)
    assert ledger.remaining(ExpansionLimitKind.TOKENS.value) >= 0
    # Recording more than remaining tokens fails closed.
    remaining_tokens = ledger.remaining(ExpansionLimitKind.TOKENS.value)
    if remaining_tokens == 0:
        with pytest.raises(ExpansionLimitExceededError):
            ledger.record(tokens=1)
    else:
        with pytest.raises(ExpansionLimitExceededError):
            ledger.record(tokens=remaining_tokens + 1)

    # 3) Concurrent calibration CAS: at most one winner, no lost_updates.
    store_dir = tmp_path / "e2e-cas"
    with DurableCoordinationStore(store_dir) as setup:
        one = _store_entry(setup, "e2e-one")
        two = _store_entry(setup, "e2e-two")

    def cal_attempt(entry_cid: str, operation_id: str) -> str:
        with DurableCoordinationStore(store_dir) as store:
            repo = DurableAuditHistoryStore(store)
            result = repo.append_calibration(
                WORKSPACE,
                entry_cid=entry_cid,
                expected_generation=0,
                expected_head_cid=None,
                operation_id=operation_id,
            )
            return result.status.value

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = [
            future.result()
            for future in as_completed(
                [
                    pool.submit(cal_attempt, one, "e2e-w1"),
                    pool.submit(cal_attempt, two, "e2e-w2"),
                ]
            )
        ]
    assert sorted(statuses) == ["conflict", "updated"]

    with DurableCoordinationStore(store_dir) as store:
        report = recover_governor_store(store, rebuild=True)
        assert report.errors == ()
        head = DurableAuditHistoryStore(store).current_history(WORKSPACE, "calibration")
        assert head.generation == 1

    # 4) Disclosure and quality claim still fail closed after recovery work.
    with pytest.raises(DisclosureForbiddenError):
        prepare_provider_invocation(
            _private_context(),
            default_shadow_disclosure_policy(),
            provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
            worktree_id="worktree-e2e-1",
        )
    with pytest.raises(SimulatedLiveQualityError):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            metadata={"live_quality": True},
        )
    with pytest.raises(SecretAdmissionError):
        project_public_report(
            {
                "summary": "post-recovery public report",
                "api_key": CANARY_API_KEY,
            }
        )

    # 5) Cancellation still leaves production untouched after the composed path.
    fingerprint = production_fingerprint_from_refs(
        repository_state_cid=_cid("repo-state"),
        head_commit="e2e-head",
    )
    guard = ProductionCheckoutGuard(fingerprint=fingerprint)
    cancelled = execute_shadow_plan(
        _shadow_plan(),
        attempt_runner=SimulatedShadowAttemptRunner(),
        production_guard=guard,
        cancellation_token=ShadowCancellationToken(cancelled=True),
    )
    assert production_state_unchanged(guard) is True
    assert cancelled.compressed_attempt.attempt_status == (
        AttemptTerminalStatus.CANCELLED.value
    )
    assert (
        cancelled.compressed_attempt.acceptance_disposition
        != AcceptanceDisposition.ACCEPTED.value
    )


def test_no_silent_overwrite_disclosure_or_quality_claim_matrix() -> None:
    """Compact matrix of the three silent-failure classes that must fail closed."""

    # Overwrite: stale generation expectation is a typed conflict, not success.
    # (In-memory CAS analogue using the policy repository is covered durably above;
    # here we assert the closed status vocabulary includes CONFLICT.)
    assert GovernorStoreStatus.CONFLICT.value == "conflict"
    assert GovernorStoreStatus.UPDATED.value == "updated"
    assert GovernorStoreStatus.CONFLICT is not GovernorStoreStatus.UPDATED

    # Disclosure: forbidden is never "allowed".
    auth = authorize_shadow_disclosure(
        default_shadow_disclosure_policy(),
        provider_id="external.unknown.provider",
        context=_private_context(),
        worktree_id="worktree-matrix-1",
        raise_on_forbidden=False,
    )
    assert auth.allowed is False
    assert auth.disposition == DisclosureDisposition.FORBIDDEN.value
    assert auth.strip_private_source is True

    # Quality claim: simulated + live claim fails; simulated alone does not apply.
    with pytest.raises(SimulatedLiveQualityError):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            metadata={"claims_live_quality": True},
        )
    # Explicit non-claim path remains non-raising.
    reject_simulated_live_quality_claim(
        execution_mode=ExecutionMode.SIMULATED.value,
        metadata={"calibration_only": True},
    )
