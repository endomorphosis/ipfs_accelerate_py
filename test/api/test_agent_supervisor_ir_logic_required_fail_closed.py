"""DCR-035: enforce mandatory logic stages and fail closed on unknown.

Acceptance
----------
* Empty surfaces cannot pass or grant execution.
* Skipped stages cannot pass or grant execution.
* Unsupported semantics cannot pass or grant execution.
* Import failures cannot pass or grant execution.
* UI bridge-only projections cannot pass or grant execution.
* Unknown / error rows block pass; no-false-grant is sealed.
* Partial-stage pass, exception swallowing, and default-true safety claims
  are forbidden.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.ir_logic_hooks import (
    IR_LOGIC_HOOKS_INTERFACE,
    IRLogicHookError,
    IRLogicHooks,
    bridge_only_observation,
    build_stage_map,
    consult_ir_logic_gate,
    default_ir_logic_hooks,
    empty_surface_observation,
    error_observation,
    import_failure_observation,
    production_pass_observation,
    skipped_stage_observation,
    unknown_observation,
    unsupported_semantics_observation,
)
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    CONTRACT_VERSION,
    DCR_ARTIFACT_PATH,
    DCR_TASK_ID,
    GateDisposition,
    IR_APPLICATION_FAILED,
    IR_APPLICATION_RESULT_INTERFACE,
    IRApplicationResult,
    IRLogicApplicationError,
    LogicDecisionKind,
    LogicStageId,
    LogicStageObservation,
    POLICY_DECISION_KINDS,
    REQUIRED_LOGIC_STAGES,
    REQUIRED_LOGIC_STAGE_GATE_INTERFACE,
    RequiredLogicStageGate,
    StageDisposition,
    SurfaceKind,
    UNKNOWN_GATE_EVIDENCE_TERM,
    all_stages_passed,
    apply_ir_logic,
    ensure_logic_gate_artifact,
    load_logic_gate,
    materialize_logic_gate_artifact,
    require_gate_pass,
    stage_observation,
    write_logic_gate,
)


# ---------------------------------------------------------------------------
# Interfaces / symbols
# ---------------------------------------------------------------------------


def test_interfaces_and_symbols() -> None:
    assert REQUIRED_LOGIC_STAGE_GATE_INTERFACE == "RequiredLogicStageGate@1"
    assert IR_APPLICATION_RESULT_INTERFACE == "IRApplicationResult@1"
    assert IR_LOGIC_HOOKS_INTERFACE == "IRLogicHooks@1"
    assert UNKNOWN_GATE_EVIDENCE_TERM == "dcr/unknown-gate@1"
    assert IR_APPLICATION_FAILED == "IR_APPLICATION_FAILED"
    assert CONTRACT_VERSION == 1
    assert DCR_TASK_ID == "DCR-035"
    assert DCR_ARTIFACT_PATH.endswith("logic-gate.json")
    assert RequiredLogicStageGate.INTERFACE == REQUIRED_LOGIC_STAGE_GATE_INTERFACE
    assert IRApplicationResult.INTERFACE == IR_APPLICATION_RESULT_INTERFACE
    assert IRLogicHooks.INTERFACE == IR_LOGIC_HOOKS_INTERFACE
    assert tuple(REQUIRED_LOGIC_STAGES) == (
        "normalize",
        "obligate",
        "route",
        "reconstruct",
        "cache",
    )
    assert tuple(POLICY_DECISION_KINDS) == (
        "diagnose",
        "plan",
        "admit",
        "apply",
        "complete",
    )
    assert callable(apply_ir_logic)
    assert callable(RequiredLogicStageGate)
    assert callable(default_ir_logic_hooks)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_full_pass_allows_decision_without_default_execution_grant() -> None:
    observations = all_stages_passed()
    for decision in POLICY_DECISION_KINDS:
        result = apply_ir_logic(decision, observations, claim_execution=False)
        assert result.gate_passed is True
        assert result.execution_granted is False
        assert result.no_false_grant is True
        assert result.disposition is GateDisposition.PASSED
        assert set(result.required_stages) == set(REQUIRED_LOGIC_STAGES)
        assert set(result.ran_stages) == set(REQUIRED_LOGIC_STAGES)
        assert set(result.pass_stages) == set(REQUIRED_LOGIC_STAGES)
        assert result.unknown_rows == ()
        assert result.unsupported_rows == ()
        assert result.error_rows == ()
        assert result.model_calls == 0
        assert result.evidence_term == UNKNOWN_GATE_EVIDENCE_TERM
        assert result.verifies_identity() is True


def test_apply_and_complete_grant_only_with_explicit_claim() -> None:
    observations = all_stages_passed()

    apply_no_claim = apply_ir_logic(
        LogicDecisionKind.APPLY, observations, claim_execution=False
    )
    assert apply_no_claim.gate_passed is True
    assert apply_no_claim.execution_granted is False

    apply_claim = apply_ir_logic(
        LogicDecisionKind.APPLY, observations, claim_execution=True
    )
    assert apply_claim.gate_passed is True
    assert apply_claim.execution_granted is True
    assert apply_claim.no_false_grant is True

    complete_claim = apply_ir_logic(
        LogicDecisionKind.COMPLETE, observations, claim_execution=True
    )
    assert complete_claim.execution_granted is True

    # Non-execution decisions never grant even with claim_execution=True.
    diagnose = apply_ir_logic(
        LogicDecisionKind.DIAGNOSE, observations, claim_execution=True
    )
    assert diagnose.gate_passed is True
    assert diagnose.execution_granted is False


# ---------------------------------------------------------------------------
# Fail-closed acceptance matrix
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "decision",
    list(POLICY_DECISION_KINDS),
)
def test_empty_surface_cannot_pass_or_grant(decision: str) -> None:
    observations = build_stage_map(
        {
            "normalize": empty_surface_observation(
                LogicStageId.NORMALIZE, family="contract_graph"
            )
        }
    )
    result = apply_ir_logic(decision, observations, claim_execution=True)
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert result.no_false_grant is True
    assert IR_APPLICATION_FAILED in result.reason_codes
    assert any("empty_surface" in code for code in result.reason_codes)
    assert result.error_rows
    assert "normalize" not in result.pass_stages


@pytest.mark.parametrize("decision", list(POLICY_DECISION_KINDS))
def test_skipped_stage_cannot_pass_or_grant(decision: str) -> None:
    observations = build_stage_map(
        {"route": skipped_stage_observation(LogicStageId.ROUTE)}
    )
    result = apply_ir_logic(decision, observations, claim_execution=True)
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert any("skipped_stage" in code for code in result.reason_codes)
    assert "route" not in result.pass_stages


@pytest.mark.parametrize("decision", list(POLICY_DECISION_KINDS))
def test_unsupported_semantics_cannot_pass_or_grant(decision: str) -> None:
    observations = build_stage_map(
        {
            "obligate": unsupported_semantics_observation(
                LogicStageId.OBLIGATE, family="profile_x"
            )
        }
    )
    result = apply_ir_logic(decision, observations, claim_execution=True)
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert result.unsupported_rows
    assert any("unsupported_semantics" in code for code in result.reason_codes)
    assert result.disposition is GateDisposition.BLOCKED


@pytest.mark.parametrize("decision", list(POLICY_DECISION_KINDS))
def test_import_failure_cannot_pass_or_grant(decision: str) -> None:
    observations = build_stage_map(
        {
            "cache": import_failure_observation(
                LogicStageId.CACHE,
                module_origin="ipfs_datasets_py.logic.missing",
            )
        }
    )
    result = apply_ir_logic(decision, observations, claim_execution=True)
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert result.error_rows
    assert any("import_failure" in code for code in result.reason_codes)


@pytest.mark.parametrize("decision", list(POLICY_DECISION_KINDS))
def test_ui_bridge_only_projection_cannot_pass_or_grant(decision: str) -> None:
    observations = build_stage_map(
        {
            "normalize": bridge_only_observation(
                LogicStageId.NORMALIZE, family="ui_ir"
            )
        }
    )
    result = apply_ir_logic(decision, observations, claim_execution=True)
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert any("bridge_only" in code for code in result.reason_codes)
    assert result.error_rows
    assert any(
        row.get("reason") == "ui_bridge_only_projection"
        for row in result.error_rows
    )


def test_unknown_outcome_blocks_pass_and_grant() -> None:
    observations = build_stage_map(
        {"reconstruct": unknown_observation(LogicStageId.RECONSTRUCT)}
    )
    result = apply_ir_logic(
        LogicDecisionKind.ADMIT, observations, claim_execution=True
    )
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert result.unknown_rows
    assert result.disposition is GateDisposition.BLOCKED
    assert any("unknown" in code for code in result.reason_codes)


def test_error_and_exception_swallowed_cannot_pass() -> None:
    observations = build_stage_map(
        {
            "route": error_observation(
                LogicStageId.ROUTE,
                detail="solver exploded",
                exception_swallowed=True,
            )
        }
    )
    result = apply_ir_logic(
        LogicDecisionKind.PLAN, observations, claim_execution=True
    )
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert result.error_rows
    # Swallowed exceptions are forced to ERROR disposition.
    assert any(
        obs.disposition is StageDisposition.ERROR for obs in result.observations
    )


def test_missing_required_stage_fails_closed() -> None:
    # Only four of five required stages present.
    observations = tuple(
        production_pass_observation(stage)
        for stage in REQUIRED_LOGIC_STAGES
        if stage != "cache"
    )
    result = apply_ir_logic(
        LogicDecisionKind.COMPLETE, observations, claim_execution=True
    )
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert "cache" not in result.ran_stages
    assert any("missing_stage:cache" in code for code in result.reason_codes)


def test_partial_stage_cannot_pass() -> None:
    observations = build_stage_map(
        {
            "obligate": stage_observation(
                LogicStageId.OBLIGATE,
                StageDisposition.PARTIAL,
                reason_codes=("partial_stage",),
            )
        }
    )
    result = apply_ir_logic(LogicDecisionKind.DIAGNOSE, observations)
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert any("partial" in code for code in result.reason_codes)


def test_bridge_only_claimed_as_pass_is_demoted() -> None:
    obs = LogicStageObservation(
        stage=LogicStageId.NORMALIZE,
        disposition=StageDisposition.PASS,
        surface_kind=SurfaceKind.BRIDGE_ONLY,
        reason_codes=("pretend_pass",),
        family="ui_ir",
    )
    assert obs.disposition is StageDisposition.BRIDGE_ONLY
    observations = build_stage_map({"normalize": obs})
    result = apply_ir_logic(LogicDecisionKind.APPLY, observations, claim_execution=True)
    assert result.gate_passed is False
    assert result.execution_granted is False


def test_empty_surface_claimed_as_pass_is_demoted() -> None:
    obs = LogicStageObservation(
        stage=LogicStageId.ROUTE,
        disposition=StageDisposition.PASS,
        surface_kind=SurfaceKind.EMPTY,
    )
    assert obs.disposition is StageDisposition.EMPTY_SURFACE
    result = apply_ir_logic(
        LogicDecisionKind.ADMIT,
        build_stage_map({"route": obs}),
        claim_execution=True,
    )
    assert result.gate_passed is False
    assert result.execution_granted is False


def test_spurious_execution_claim_without_full_pass_is_rejected() -> None:
    observations = build_stage_map(
        {
            "cache": stage_observation(
                LogicStageId.CACHE,
                StageDisposition.FAIL,
                grants_execution_claim=True,
                reason_codes=("fake_grant",),
            )
        }
    )
    result = apply_ir_logic(
        LogicDecisionKind.APPLY, observations, claim_execution=True
    )
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert "spurious_execution_claim" in result.reason_codes or (
        IR_APPLICATION_FAILED in result.reason_codes
    )


def test_execution_cannot_be_constructed_without_gate_pass() -> None:
    with pytest.raises(IRLogicApplicationError, match="execution cannot be granted"):
        IRApplicationResult(
            decision=LogicDecisionKind.APPLY,
            disposition=GateDisposition.FAILED,
            required_stages=REQUIRED_LOGIC_STAGES,
            ran_stages=(),
            pass_stages=(),
            gate_passed=False,
            execution_granted=True,
            no_false_grant=True,
        )


def test_gate_passed_requires_full_pass_set() -> None:
    with pytest.raises(IRLogicApplicationError, match="pass_stages"):
        IRApplicationResult(
            decision=LogicDecisionKind.PLAN,
            disposition=GateDisposition.PASSED,
            required_stages=REQUIRED_LOGIC_STAGES,
            ran_stages=REQUIRED_LOGIC_STAGES,
            pass_stages=("normalize",),
            gate_passed=True,
            execution_granted=False,
            no_false_grant=True,
        )


def test_model_calls_must_be_zero() -> None:
    with pytest.raises(IRLogicApplicationError, match="model_calls"):
        IRApplicationResult(
            decision=LogicDecisionKind.PLAN,
            disposition=GateDisposition.FAILED,
            required_stages=REQUIRED_LOGIC_STAGES,
            ran_stages=(),
            pass_stages=(),
            model_calls=1,
        )


def test_required_stages_cannot_be_shrunk() -> None:
    with pytest.raises(IRLogicApplicationError, match="missing mandatory"):
        RequiredLogicStageGate(required_stages=("normalize", "cache"))


def test_unknown_required_stage_rejected() -> None:
    with pytest.raises(IRLogicApplicationError, match="unknown stages"):
        RequiredLogicStageGate(
            required_stages=REQUIRED_LOGIC_STAGES + ("invented",)
        )


# ---------------------------------------------------------------------------
# Planning hooks
# ---------------------------------------------------------------------------


def test_hooks_pass_for_full_production_observations() -> None:
    hooks = default_ir_logic_hooks(raise_on_failure=True)
    observations = all_stages_passed()
    for method_name in ("diagnose", "plan", "admit"):
        method = getattr(hooks, method_name)
        result = method(observations)
        assert result.gate_passed is True
        assert result.execution_granted is False


def test_hooks_apply_complete_require_explicit_execution_claim() -> None:
    hooks = IRLogicHooks(raise_on_failure=True)
    observations = all_stages_passed()

    applied = hooks.apply(observations, claim_execution=False)
    assert applied.gate_passed is True
    assert applied.execution_granted is False

    granted = hooks.require_execution(LogicDecisionKind.APPLY, observations)
    assert granted.execution_granted is True

    completed = hooks.require_execution(LogicDecisionKind.COMPLETE, observations)
    assert completed.execution_granted is True


def test_hooks_raise_on_empty_surface() -> None:
    hooks = default_ir_logic_hooks(raise_on_failure=True)
    observations = build_stage_map(
        {"normalize": empty_surface_observation(LogicStageId.NORMALIZE)}
    )
    with pytest.raises(IRLogicHookError) as exc_info:
        hooks.plan(observations)
    assert exc_info.value.reason_code == IR_APPLICATION_FAILED


def test_hooks_raise_on_skipped_stage() -> None:
    hooks = IRLogicHooks(raise_on_failure=True)
    observations = build_stage_map(
        {"obligate": skipped_stage_observation(LogicStageId.OBLIGATE)}
    )
    with pytest.raises(IRLogicHookError, match="IR_APPLICATION_FAILED|failed|skipped"):
        hooks.admit(observations)


def test_hooks_raise_on_unsupported_semantics() -> None:
    hooks = IRLogicHooks(raise_on_failure=True)
    observations = build_stage_map(
        {
            "route": unsupported_semantics_observation(
                LogicStageId.ROUTE, family="remote_llm"
            )
        }
    )
    with pytest.raises(IRLogicHookError):
        hooks.diagnose(observations)


def test_hooks_raise_on_import_failure() -> None:
    hooks = IRLogicHooks(raise_on_failure=True)
    observations = build_stage_map(
        {
            "reconstruct": import_failure_observation(
                LogicStageId.RECONSTRUCT,
                module_origin="ipfs_datasets_py.logic.backends.missing",
            )
        }
    )
    with pytest.raises(IRLogicHookError):
        hooks.complete(observations, claim_execution=True)


def test_hooks_raise_on_bridge_only() -> None:
    hooks = IRLogicHooks(raise_on_failure=True)
    observations = build_stage_map(
        {"normalize": bridge_only_observation(LogicStageId.NORMALIZE)}
    )
    with pytest.raises(IRLogicHookError):
        hooks.apply(observations, claim_execution=True)


def test_hooks_can_return_failure_without_raising() -> None:
    hooks = IRLogicHooks(raise_on_failure=False)
    observations = build_stage_map(
        {"cache": unknown_observation(LogicStageId.CACHE)}
    )
    result = hooks.plan(observations)
    assert result.gate_passed is False
    assert result.execution_granted is False
    assert result.no_false_grant is True
    assert hooks.history
    assert hooks.history[-1].gate_passed is False


def test_hooks_reject_execution_for_non_execution_decision() -> None:
    hooks = IRLogicHooks(raise_on_failure=True)
    with pytest.raises(IRLogicHookError, match="cannot grant execution"):
        hooks.require_execution(LogicDecisionKind.PLAN, all_stages_passed())


def test_consult_ir_logic_gate_module_helper() -> None:
    result = consult_ir_logic_gate(
        "admit",
        all_stages_passed(),
        claim_execution=False,
        raise_on_failure=True,
    )
    assert result.gate_passed is True
    assert result.decision is LogicDecisionKind.ADMIT


def test_require_gate_pass_raises_on_failure() -> None:
    result = apply_ir_logic(
        LogicDecisionKind.DIAGNOSE,
        build_stage_map(
            {"normalize": empty_surface_observation(LogicStageId.NORMALIZE)}
        ),
    )
    with pytest.raises(IRLogicApplicationError, match="IR_APPLICATION_FAILED|failed"):
        require_gate_pass(result)


def test_require_gate_pass_requires_execution_when_asked() -> None:
    result = apply_ir_logic(
        LogicDecisionKind.APPLY,
        all_stages_passed(),
        claim_execution=False,
    )
    assert result.gate_passed is True
    with pytest.raises(IRLogicApplicationError, match="execution was not granted"):
        require_gate_pass(result, require_execution=True)


# ---------------------------------------------------------------------------
# Evidence / artifact
# ---------------------------------------------------------------------------


def test_result_identity_is_content_addressed() -> None:
    first = apply_ir_logic(LogicDecisionKind.PLAN, all_stages_passed())
    second = apply_ir_logic(LogicDecisionKind.PLAN, all_stages_passed())
    assert first.result_id == second.result_id
    assert first.canonical_digest == second.canonical_digest
    assert first.verifies_identity() is True
    payload = first.to_dict()
    reloaded = IRApplicationResult.from_dict(payload)
    assert reloaded.result_id == first.result_id


def test_materialize_and_write_logic_gate_artifact(tmp_path: Path) -> None:
    result = apply_ir_logic(LogicDecisionKind.DIAGNOSE, all_stages_passed())
    artifact = materialize_logic_gate_artifact(results=(result,))
    assert artifact["schema"].endswith("logic-gate-artifact@1")
    assert artifact["interface"] == REQUIRED_LOGIC_STAGE_GATE_INTERFACE
    assert artifact["evidence_term"] == UNKNOWN_GATE_EVIDENCE_TERM
    assert artifact["task_id"] == "DCR-035"
    assert artifact["acceptance"]["no_false_grant"] is True
    assert artifact["acceptance"]["empty_surfaces_cannot_pass_or_grant"] is True
    assert artifact["acceptance"]["skipped_stages_cannot_pass_or_grant"] is True
    assert (
        artifact["acceptance"]["unsupported_semantics_cannot_pass_or_grant"]
        is True
    )
    assert artifact["acceptance"]["import_failures_cannot_pass_or_grant"] is True
    assert (
        artifact["acceptance"]["ui_bridge_only_projections_cannot_pass_or_grant"]
        is True
    )
    assert artifact["required_stages"] == list(REQUIRED_LOGIC_STAGES)

    path = write_logic_gate(tmp_path / "logic-gate.json", artifact=artifact)
    assert path.is_file()
    loaded = load_logic_gate(path)
    assert loaded["task_id"] == "DCR-035"
    assert loaded["acceptance"]["no_false_grant"] is True


def test_ensure_logic_gate_artifact_roundtrip(tmp_path: Path) -> None:
    # Point repo_root at tmp so we do not touch the real workspace artifact.
    dest = tmp_path / "data/agent_supervisor/deterministic_contract_repair"
    dest.mkdir(parents=True)
    # write via ensure against a custom root by writing first then loading
    path = write_logic_gate(
        dest / "logic-gate.json",
        results=(
            apply_ir_logic(LogicDecisionKind.ADMIT, all_stages_passed()),
        ),
    )
    loaded = load_logic_gate(path)
    assert loaded["interface"] == REQUIRED_LOGIC_STAGE_GATE_INTERFACE
    # ensure with force rewrites
    rewritten = ensure_logic_gate_artifact(repo_root=tmp_path, force=True)
    assert rewritten.is_file()
    body = json.loads(rewritten.read_text(encoding="utf-8"))
    assert body["evidence_term"] == UNKNOWN_GATE_EVIDENCE_TERM


def test_load_logic_gate_rejects_missing_no_false_grant(tmp_path: Path) -> None:
    path = tmp_path / "bad-logic-gate.json"
    path.write_text(
        json.dumps(
            {
                "schema": "ipfs_accelerate_py/agent-supervisor/logic-gate-artifact@1",
                "interface": REQUIRED_LOGIC_STAGE_GATE_INTERFACE,
                "evidence_term": UNKNOWN_GATE_EVIDENCE_TERM,
                "acceptance": {"no_false_grant": False},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(IRLogicApplicationError, match="no_false_grant"):
        load_logic_gate(path)


def test_hooks_to_dict_seals_acceptance() -> None:
    hooks = default_ir_logic_hooks(raise_on_failure=False)
    hooks.diagnose(all_stages_passed())
    payload = hooks.to_dict()
    assert payload["interface"] == IR_LOGIC_HOOKS_INTERFACE
    assert payload["acceptance"]["exception_swallowing_forbidden"] is True
    assert payload["acceptance"]["default_true_safety_claims_forbidden"] is True
    assert payload["acceptance"]["no_false_grant"] is True
    assert payload["history"]


def test_evidence_subset_fields_present_on_failure() -> None:
    result = apply_ir_logic(
        LogicDecisionKind.APPLY,
        build_stage_map(
            {
                "normalize": empty_surface_observation(LogicStageId.NORMALIZE),
                "route": unsupported_semantics_observation(LogicStageId.ROUTE),
                "cache": unknown_observation(LogicStageId.CACHE),
            }
        ),
        claim_execution=True,
    )
    # Evidence subset: required/ran/pass sets, unknown/unsupported/error rows,
    # no-false-grant claim.
    assert result.required_stages
    assert isinstance(result.ran_stages, tuple)
    assert isinstance(result.pass_stages, tuple)
    assert result.unknown_rows
    assert result.unsupported_rows
    assert result.error_rows
    assert result.no_false_grant is True
    assert result.execution_granted is False
    assert result.gate_passed is False


def test_all_decision_kinds_covered_by_hooks() -> None:
    hooks = IRLogicHooks(raise_on_failure=False)
    observations = all_stages_passed()
    results = [
        hooks.diagnose(observations),
        hooks.plan(observations),
        hooks.admit(observations),
        hooks.apply(observations, claim_execution=True),
        hooks.complete(observations, claim_execution=True),
    ]
    assert all(item.gate_passed for item in results)
    assert results[0].execution_granted is False
    assert results[1].execution_granted is False
    assert results[2].execution_granted is False
    assert results[3].execution_granted is True
    assert results[4].execution_granted is True
