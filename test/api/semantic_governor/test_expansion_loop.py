"""Tests for SCG-029 bounded counterexample-guided expansion loop.

Acceptance criteria enforced here:

* Limits are enforced across restart (spent counters restore from checkpoint).
* Supported omission can repair before frontier escalation.
* Both-context failure can request route escalation without blaming compression.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_governor.audit_contracts import (
    ContextExpansionPlan,
    ContextExpansionStep,
    DecisionAction,
    ExpansionAction,
    ExpansionStepStatus,
    RouteTier,
)
from ipfs_datasets_py.logic.software_contracts.semantic_governor.base import (
    ArtifactProvenance,
    AssumptionKind,
    AuthoritySource,
    ContextSufficiencyState,
    ExecutionMode,
    GeneratorIdentity,
    GovernorArtifactHeader,
    GovernorAssumption,
    GovernorTerminalStatus,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.contracts import (
    ComparativeOutcome,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.expansion_loop import (
    EXECUTE_EXPANSION_LOOP_INTERFACE,
    EXPANSION_LOOP_RESULT_SCHEMA,
    SCG_EXPANSION_LOOP_EVIDENCE,
    AlwaysFailRunner,
    ExpansionBudgetLedger,
    ExpansionLimitExceededError,
    ExpansionLimitKind,
    ExpansionLoopCheckpoint,
    ExpansionLoopDisposition,
    ExpansionLoopError,
    ExpansionLoopResult,
    FilesystemExpansionCheckpointStore,
    InMemoryExpansionCheckpointStore,
    RepairingOnArtifactRunner,
    ScriptedExpansionStepRunner,
    both_context_failure_outcomes,
    compression_blame_reason_codes,
    default_model_policy,
    default_verification_policy,
    execute_expansion_loop,
    execute_expansion_loop_interface_id,
    omission_supporting_outcomes,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT
    / "ipfs_accelerate_py/agent_supervisor/semantic_governor/expansion_loop.py"
)


# ---------------------------------------------------------------------------
# Fixtures / recipes
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
            generator_id="expansion_planner",
            generator_version="1.0.0",
            interface_id="plan_context_expansion@1",
        ),
        "provenance": ArtifactProvenance(
            producer_id="semantic_governor",
            producer_version="1",
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=(_cid("input-a"),),
            tool_ids=("expansion.v1",),
            policy_cid=_cid("policy"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="expansion_bounded",
                kind=AssumptionKind.BUDGET,
                statement="Context expansion is hard-bounded",
                supporting_cids=(_cid("budget"),),
            ),
        ),
        "metadata": {"task": "SCG-029"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)


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
        "changed_assumption_ids": ("expansion_bounded",),
        "hypothesis_supported": None,
        "notes": None,
        "metadata": {},
    }
    fields.update(overrides)
    return ContextExpansionStep(**fields)


def _plan(
    steps: list[ContextExpansionStep] | None = None,
    **overrides: Any,
) -> ContextExpansionPlan:
    if steps is None:
        steps = [_step()]
    total = sum(s.token_increase for s in steps)
    fields: dict[str, Any] = {
        "header": _header("context_expansion_plan"),
        "plan_id": "plan_scg029",
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
    # Keep totals consistent when overrides change steps without totals.
    if "steps" in overrides and "total_token_increase" not in overrides:
        resolved_steps = tuple(fields["steps"])
        fields["total_token_increase"] = sum(s.token_increase for s in resolved_steps)
        fields["step_count"] = len(resolved_steps)
    return ContextExpansionPlan(**fields)


# ---------------------------------------------------------------------------
# Structural / contract tests
# ---------------------------------------------------------------------------


def test_module_exists_and_exports_interface() -> None:
    assert MODULE_PATH.is_file()
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "execute_expansion_loop" in names
    assert execute_expansion_loop_interface_id() == EXECUTE_EXPANSION_LOOP_INTERFACE
    assert SCG_EXPANSION_LOOP_EVIDENCE == "scg/expansion-loop@1"


def test_evidence_and_closed_vocabularies() -> None:
    assert ComparativeOutcome.BOTH_FAILED_SAME_REASON.value in both_context_failure_outcomes()
    assert (
        ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value
        in omission_supporting_outcomes()
    )
    blame = set(compression_blame_reason_codes())
    assert "compression_omission" in blame
    assert "omission_blame" in blame


# ---------------------------------------------------------------------------
# Acceptance: supported omission can repair before frontier
# ---------------------------------------------------------------------------


def test_supported_omission_repairs_before_frontier() -> None:
    """Context expansion that supplies the missing artifact repairs without escalation."""

    context = _step(
        step_id="step_0000_include_raw_source_helper",
        step_index=0,
        action=ExpansionAction.INCLUDE_RAW_SOURCE.value,
        token_increase=120,
        artifact_ids_added=("exc_helper",),
    )
    escalate = _step(
        step_id="step_0001_escalate_route_model",
        step_index=1,
        action=ExpansionAction.ESCALATE_ROUTE.value,
        token_increase=0,
        artifact_ids_added=(),
        reason_code="model_route_after_context",
    )
    plan = _plan(steps=[context, escalate], max_token_growth=500)

    result = execute_expansion_loop(
        plan,
        default_model_policy(),
        default_verification_policy(),
        runner=RepairingOnArtifactRunner(required_artifact_id="exc_helper"),
        comparative_outcome=ComparativeOutcome.COMPRESSED_FAILED_EXPANDED_SUCCEEDED.value,
        counterexample_cids=(_cid("cex-1"),),
    )

    assert result.disposition == ExpansionLoopDisposition.REPAIRED.value
    assert result.repaired is True
    assert result.frontier_escalation_requested is False
    assert result.compression_blamed is False
    assert result.context_before_model_escalation is True
    assert result.decision_action == DecisionAction.RETRY_SAME_ROUTE.value
    assert result.route_tier == RouteTier.MEDIUM.value
    assert "supported_omission_repaired_before_frontier" in result.reason_codes
    assert "exc_helper" in result.artifacts_included
    # Escalate step must not have been applied.
    assert all(e.action != ExpansionAction.ESCALATE_ROUTE.value for e in result.executed_steps)
    assert result.executed_steps[0].hypothesis_supported is True
    assert result.budget["spent_tokens"] == 120
    assert result.budget["spent_escalations"] == 0
    # Result is content-addressed and round-trips.
    restored = ExpansionLoopResult.from_dict(result.to_dict())
    assert restored.result_cid == result.result_cid
    assert result.to_dict()["schema"] == EXPANSION_LOOP_RESULT_SCHEMA


def test_context_steps_precede_escalation_in_execution_order() -> None:
    """Even when both fail, planned context steps run before escalate_route."""

    context = _step(
        step_id="step_0000_include_raw_source_helper",
        step_index=0,
        artifact_ids_added=("exc_helper",),
        token_increase=50,
    )
    escalate = _step(
        step_id="step_0001_escalate_route_model",
        step_index=1,
        action=ExpansionAction.ESCALATE_ROUTE.value,
        token_increase=0,
        artifact_ids_added=(),
        reason_code="model_route_after_context",
    )
    plan = _plan(steps=[context, escalate], max_token_growth=200)

    result = execute_expansion_loop(
        plan,
        default_model_policy(),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        comparative_outcome=ComparativeOutcome.BOTH_FAILED_SAME_REASON.value,
    )

    assert result.frontier_escalation_requested is True
    assert result.disposition == ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value
    actions = [e.action for e in result.executed_steps]
    assert ExpansionAction.INCLUDE_RAW_SOURCE.value in actions
    assert ExpansionAction.ESCALATE_ROUTE.value in actions
    assert actions.index(ExpansionAction.INCLUDE_RAW_SOURCE.value) < actions.index(
        ExpansionAction.ESCALATE_ROUTE.value
    )


# ---------------------------------------------------------------------------
# Acceptance: both-context failure escalates without blaming compression
# ---------------------------------------------------------------------------


def test_both_context_failure_escalates_without_blaming_compression() -> None:
    """Both-fail may request frontier escalation; compression_blamed stays false."""

    # Plan with only model-route escalation (no supported omission path).
    escalate = _step(
        step_id="step_0000_escalate_route_model",
        step_index=0,
        action=ExpansionAction.ESCALATE_ROUTE.value,
        token_increase=0,
        artifact_ids_added=(),
        reason_code="model_route_only",
    )
    plan = _plan(
        steps=[escalate],
        max_token_growth=0,
        total_token_increase=0,
        max_escalations=1,
    )

    result = execute_expansion_loop(
        plan,
        default_model_policy(allow_frontier_escalation=True),
        default_verification_policy(),
        runner=AlwaysFailRunner(),
        comparative_outcome=ComparativeOutcome.BOTH_FAILED_DIFFERENT_REASON.value,
    )

    assert result.disposition == ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value
    assert result.frontier_escalation_requested is True
    assert result.compression_blamed is False
    assert result.repaired is False
    assert result.decision_action == DecisionAction.ESCALATE_FRONTIER.value
    assert (
        result.sufficiency_state
        == ContextSufficiencyState.FRONTIER_ESCALATION_REQUIRED.value
    )
    assert result.route_tier == RouteTier.FRONTIER.value
    assert "both_context_failure" in result.reason_codes
    assert "route_escalation_without_omission_blame" in result.reason_codes
    blame = set(compression_blame_reason_codes())
    assert set(result.reason_codes).isdisjoint(blame)


def test_both_context_failure_after_failed_context_expansion_no_blame() -> None:
    """Context expansion runs, still fails both sides → escalate without omission blame."""

    context = _step(
        step_id="step_0000_strengthen_capsule_helper",
        step_index=0,
        action=ExpansionAction.STRENGTHEN_CAPSULE.value,
        token_increase=80,
        artifact_ids_added=("cap_helper",),
        reason_code="strengthen_capsule",
    )
    plan = _plan(steps=[context], max_token_growth=200, max_escalations=1)

    result = execute_expansion_loop(
        plan,
        default_model_policy(),
        default_verification_policy(),
        runner=AlwaysFailRunner(reason_codes=("reasoning_failure",)),
        comparative_outcome=ComparativeOutcome.BOTH_FAILED_SAME_REASON.value,
    )

    assert result.disposition == ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value
    assert result.frontier_escalation_requested is True
    assert result.compression_blamed is False
    assert "both_context_failure" in result.reason_codes
    assert "route_escalation_without_omission_blame" in result.reason_codes
    assert "cap_helper" in result.artifacts_included
    assert set(result.reason_codes).isdisjoint(set(compression_blame_reason_codes()))


def test_empty_plan_both_fail_escalates_without_omission_blame() -> None:
    plan = _plan(steps=[], max_token_growth=0, total_token_increase=0, step_count=0)
    result = execute_expansion_loop(
        plan,
        default_model_policy(),
        default_verification_policy(),
        comparative_outcome=ComparativeOutcome.BOTH_FAILED_SAME_REASON.value,
    )
    assert result.disposition == ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value
    assert result.compression_blamed is False
    assert "no_supported_context_expansion" in result.reason_codes


def test_result_rejects_compression_blame_flag() -> None:
    plan = _plan(steps=[])
    with pytest.raises(ExpansionLoopError, match="compression_blamed"):
        ExpansionLoopResult(
            plan_cid=plan.plan_cid,
            disposition=ExpansionLoopDisposition.ROUTE_ESCALATION_REQUESTED.value,
            decision_action=DecisionAction.ESCALATE_FRONTIER.value,
            sufficiency_state=ContextSufficiencyState.FRONTIER_ESCALATION_REQUIRED.value,
            route_tier=RouteTier.FRONTIER.value,
            repaired=False,
            frontier_escalation_requested=True,
            compression_blamed=True,
            context_before_model_escalation=True,
            reason_codes=("route_escalation_requested",),
            executed_steps=(),
            budget={"spent_steps": 0},
            model_policy_cid=default_model_policy().policy_cid,
            verification_policy_cid=default_verification_policy().policy_cid,
        )


# ---------------------------------------------------------------------------
# Acceptance: limits enforced across restart
# ---------------------------------------------------------------------------


def test_limits_enforced_across_restart_via_checkpoint(tmp_path: Path) -> None:
    """Spent counters restore from durable checkpoint; remaining limits apply."""

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
        _step(
            step_id="step_0002_include_raw_source_c",
            step_index=2,
            token_increase=40,
            artifact_ids_added=("art_c",),
        ),
    ]
    plan = _plan(
        steps=steps,
        max_steps=3,
        max_token_growth=240,  # 100+100+40 = 240; partial spend leaves remainder
        max_retries=0,
        max_escalations=0,
    )

    store = FilesystemExpansionCheckpointStore(tmp_path / "ckpts")
    model = default_model_policy(allow_frontier_escalation=False)
    verify = default_verification_policy()
    runner = AlwaysFailRunner()

    call_count = {"n": 0}

    def cancel_after_first() -> bool:
        # Cancel before the second step is admitted (after first complete).
        call_count["n"] += 1
        return call_count["n"] > 1

    first = execute_expansion_loop(
        plan,
        model,
        verify,
        runner=runner,
        checkpoint_store=store,
        cancel_requested=cancel_after_first,
    )
    assert first.disposition == ExpansionLoopDisposition.CANCELLED.value
    assert first.budget["spent_tokens"] == 100
    assert first.budget["spent_steps"] == 1

    loaded = store.load(plan.plan_cid)
    assert loaded is not None
    assert loaded.budget["spent_tokens"] == 100
    assert loaded.next_step_index == 1

    # Resume: prior spend restored; remaining tokens still hard-bounded.
    second = execute_expansion_loop(
        plan,
        model,
        verify,
        runner=runner,
        checkpoint_store=store,
        checkpoint=loaded,
    )
    assert second.budget["spent_tokens"] >= 100
    assert second.budget["spent_tokens"] <= plan.max_token_growth
    assert second.budget["spent_steps"] <= plan.max_steps
    # Prior checkpoint spend is preserved (not reset on resume).
    assert second.budget["spent_tokens"] > 100


def test_token_limit_blocks_further_expansion_after_restart() -> None:
    """After spending near the token cap, resume refuses additional growth."""

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
    plan = _plan(
        steps=steps,
        max_steps=2,
        max_token_growth=200,
        max_retries=0,
        max_escalations=0,
    )
    store = InMemoryExpansionCheckpointStore()
    model = default_model_policy(allow_frontier_escalation=False)
    verify = default_verification_policy()

    call_count = {"n": 0}

    def cancel_after_first() -> bool:
        call_count["n"] += 1
        return call_count["n"] > 1

    first = execute_expansion_loop(
        plan,
        model,
        verify,
        runner=AlwaysFailRunner(),
        checkpoint_store=store,
        cancel_requested=cancel_after_first,
    )
    assert first.budget["spent_tokens"] == 100
    ckpt = store.load(plan.plan_cid)
    assert ckpt is not None

    # Simulate restored higher spend (or concurrent charge) that leaves only 50 tokens.
    inflated_budget = dict(ckpt.budget)
    inflated_budget["spent_tokens"] = 150
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
        model,
        verify,
        runner=AlwaysFailRunner(),
        checkpoint_store=store,
        checkpoint=inflated,
    )
    # Second step needs 100 tokens but only 50 remain.
    assert second.budget["spent_tokens"] == 150
    assert second.disposition == ExpansionLoopDisposition.LIMITS_EXHAUSTED.value
    assert "tokens" in second.reason_codes or "limit_exceeded" in second.reason_codes
    assert "art_b" not in second.artifacts_included
    assert "art_a" in second.artifacts_included


def test_retry_limit_enforced_across_checkpoint_resume() -> None:
    steps = [
        _step(
            step_id="step_0000_include_raw_source_helper",
            step_index=0,
            token_increase=50,
            artifact_ids_added=("exc_helper",),
        ),
    ]
    plan = _plan(steps=steps, max_retries=1, max_token_growth=200, max_escalations=0)
    store = InMemoryExpansionCheckpointStore()
    model = default_model_policy(
        allow_same_route_retry=True, allow_frontier_escalation=False
    )

    # Script: fail, fail, succeed — but max_retries=1 allows only one retry (2 attempts).
    runner = ScriptedExpansionStepRunner(
        script={
            "step_0000_include_raw_source_helper": (
                {"status": "failed", "selected_tests_passed": False, "counterexample_present": True},
                {"status": "failed", "selected_tests_passed": False, "counterexample_present": True},
                {"status": "succeeded", "selected_tests_passed": True, "counterexample_present": False},
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

    # Resume with same exhausted retry budget must not grant extra retries.
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
    assert resumed.budget["spent_retries"] <= plan.max_retries
    assert resumed.repaired is False


def test_budget_ledger_recheck_fails_closed() -> None:
    ledger = ExpansionBudgetLedger(
        max_steps=2,
        max_token_growth=100,
        max_retries=1,
        max_escalations=1,
        max_wall_time_ms=1_000,
        max_spend_micros=1_000,
        spent_tokens=90,
    )
    with pytest.raises(ExpansionLimitExceededError) as excinfo:
        ledger.recheck(tokens=20)
    assert excinfo.value.limit_kind == ExpansionLimitKind.TOKENS.value
    ledger.record(tokens=10)
    assert ledger.snapshot()["remaining_tokens"] == 0
    with pytest.raises(ExpansionLimitExceededError):
        ledger.record(tokens=1)


# ---------------------------------------------------------------------------
# Plan ordering / policy edge cases
# ---------------------------------------------------------------------------


def test_rejects_plan_with_context_after_escalation() -> None:
    escalate = _step(
        step_id="step_0000_escalate_route_model",
        step_index=0,
        action=ExpansionAction.ESCALATE_ROUTE.value,
        token_increase=0,
        artifact_ids_added=(),
        reason_code="model_route_only",
    )
    context = _step(
        step_id="step_0001_include_raw_source_helper",
        step_index=1,
        token_increase=50,
        artifact_ids_added=("exc_helper",),
    )
    plan = _plan(steps=[escalate, context], max_token_growth=100)
    with pytest.raises(ExpansionLoopError, match="context_before_model_escalation"):
        execute_expansion_loop(
            plan,
            default_model_policy(),
            default_verification_policy(),
            runner=AlwaysFailRunner(),
        )


def test_human_review_step_halts_without_escalation() -> None:
    review = _step(
        step_id="step_0000_request_human_review",
        step_index=0,
        action=ExpansionAction.REQUEST_HUMAN_REVIEW.value,
        token_increase=0,
        artifact_ids_added=(),
        reason_code="human_review_required",
    )
    plan = _plan(steps=[review], max_token_growth=0, total_token_increase=0)
    result = execute_expansion_loop(
        plan,
        default_model_policy(),
        default_verification_policy(),
    )
    assert result.disposition == ExpansionLoopDisposition.HUMAN_REVIEW_REQUIRED.value
    assert result.requires_human_review is True
    assert result.frontier_escalation_requested is False
    assert result.decision_action == DecisionAction.REQUIRE_HUMAN_REVIEW.value


def test_checkpoint_round_trip_identity() -> None:
    plan = _plan()
    model = default_model_policy()
    verify = default_verification_policy()
    ledger = ExpansionBudgetLedger.from_plan(plan)
    ledger.record(steps=1, tokens=50, retries=1)
    ckpt = ExpansionLoopCheckpoint(
        plan_cid=plan.plan_cid,
        model_policy_cid=model.policy_cid,
        verification_policy_cid=verify.policy_cid,
        phase="context_expansion",
        next_step_index=1,
        budget=ledger.snapshot(),
        reason_codes=("context_expansion_applied",),
        compression_blamed=False,
    )
    restored = ExpansionLoopCheckpoint.from_dict(ckpt.to_dict())
    assert restored.checkpoint_cid == ckpt.checkpoint_cid
    assert restored.budget["spent_tokens"] == 50
    assert restored.budget["spent_retries"] == 1


def test_empty_plan_without_both_fail_is_no_action() -> None:
    plan = _plan(steps=[], max_token_growth=0, total_token_increase=0, step_count=0)
    result = execute_expansion_loop(
        plan,
        default_model_policy(),
        default_verification_policy(),
    )
    assert result.disposition == ExpansionLoopDisposition.NO_ACTION.value
    assert result.frontier_escalation_requested is False
    assert result.compression_blamed is False


def test_policies_accept_mapping_inputs() -> None:
    plan = _plan(
        steps=[
            _step(
                artifact_ids_added=("exc_helper",),
                token_increase=40,
            )
        ],
        max_token_growth=100,
    )
    result = execute_expansion_loop(
        plan.to_dict(),
        default_model_policy().to_dict(),
        default_verification_policy().to_dict(),
        runner=RepairingOnArtifactRunner(required_artifact_id="exc_helper"),
    )
    assert result.repaired is True
    assert result.disposition == ExpansionLoopDisposition.REPAIRED.value
