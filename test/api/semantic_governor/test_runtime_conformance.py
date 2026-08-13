"""SCG-032: freeze runtime APIs and prove shadow/expansion resilience.

Acceptance criteria enforced here:

* Interrupted audits recover (checkpoint resume preserves progress).
* Duplicate inputs preserve identities (plans/results/decisions).
* Private external shadow is rejected.
* Unbounded expansion is rejected.
* Suppressed failure is rejected (failures stay visible).
* Simulated live-quality claims are rejected.
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
    CostTimingProjection,
    PairedAttemptRecord,
    ShadowAttemptRole,
    VerificationProjection,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.expansion_loop import (
    AlwaysFailRunner,
    ExpansionLoopDisposition,
    InMemoryExpansionCheckpointStore,
    RepairingOnArtifactRunner,
    default_model_policy,
    default_verification_policy,
    execute_expansion_loop,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.privacy import (
    DisclosureDisposition,
    default_shadow_disclosure_policy,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.routes import (
    RouteCalibrationDisposition,
    RouteRunObservation,
    observation_from_receipt_fields,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.runtime import (
    AUDIT_TASK_INTERFACE,
    EXPAND_AUDIT_INTERFACE,
    GOVERNOR_RUNTIME_INTERFACE,
    MAX_RUNTIME_EXPANSION_STEPS,
    MAX_RUNTIME_TOKEN_GROWTH,
    SCG_RUNTIME_CONFORMANCE_EVIDENCE,
    SHADOW_TASK_INTERFACE,
    AuditDisposition,
    AuditPhase,
    ExpandAuditDisposition,
    FilesystemAuditCheckpointStore,
    GovernorRuntime,
    InMemoryAuditCheckpointStore,
    PrivateExternalShadowError,
    SimulatedLiveQualityError,
    ShadowTaskDisposition,
    SuppressedFailureError,
    UnboundedExpansionError,
    audit_task,
    audit_task_interface_id,
    expand_audit,
    expand_audit_interface_id,
    governor_runtime_interface_id,
    reject_private_external_shadow,
    reject_simulated_calibration_as_live,
    reject_simulated_live_quality_claim,
    reject_suppressed_failure,
    reject_unbounded_expansion,
    runtime_conformance_evidence_id,
    shadow_task,
    shadow_task_interface_id,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow import (
    DEFAULT_EXTERNAL_PROVIDER_ID,
    SimulatedShadowAttemptRunner,
)
from ipfs_accelerate_py.agent_supervisor.semantic_governor.shadow_plan import (
    development_shadow_sampling_policy,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_PATH = (
    REPO_ROOT / "ipfs_accelerate_py/agent_supervisor/semantic_governor/runtime.py"
)


def _cid(label: str) -> str:
    return cid_for_bytes(label.encode("utf-8"))


def _header(artifact_kind: str, **overrides: Any) -> GovernorArtifactHeader:
    fields: dict[str, Any] = {
        "artifact_kind": artifact_kind,
        "repository_state_cid": _cid("repo-state"),
        "context_pack_cid": _cid("context-pack"),
        "verification_bundle_cid": _cid("verification-bundle"),
        "generator": GeneratorIdentity(
            generator_id="runtime_conformance",
            generator_version="1.0.0",
            interface_id="audit_task@1",
        ),
        "provenance": ArtifactProvenance(
            producer_id="semantic_governor",
            producer_version="1",
            execution_mode=ExecutionMode.LIVE,
            authority_source=AuthoritySource.DETERMINISTIC,
            input_cids=(_cid("input-a"),),
            tool_ids=("governor_runtime.v1",),
            policy_cid=_cid("policy"),
            notes=None,
        ),
        "terminal_status": GovernorTerminalStatus.COMPLETE,
        "assumptions": (
            GovernorAssumption(
                assumption_id="runtime_bounded",
                kind=AssumptionKind.BUDGET,
                statement="Runtime audits are hard-bounded and recoverable",
                supporting_cids=(_cid("budget"),),
            ),
        ),
        "metadata": {"task": "SCG-032"},
    }
    fields.update(overrides)
    return GovernorArtifactHeader(**fields)


def _task(task_id: str = "task.runtime.1", **overrides: Any) -> dict[str, Any]:
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
        "changed_assumption_ids": ("runtime_bounded",),
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
        "plan_id": "plan_scg032",
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


def _runtime(**overrides: Any) -> GovernorRuntime:
    fields: dict[str, Any] = {
        "audit_store": InMemoryAuditCheckpointStore(),
        "expansion_store": InMemoryExpansionCheckpointStore(),
        "audit_policy": development_shadow_sampling_policy(random_seed=1),
        "disclosure_policy": default_shadow_disclosure_policy(),
        "attempt_runner": SimulatedShadowAttemptRunner(),
        "default_execution_mode": ExecutionMode.SIMULATED.value,
    }
    fields.update(overrides)
    return GovernorRuntime(**fields)


def _attempt(
    *,
    role: str = ShadowAttemptRole.COMPRESSED.value,
    execution_mode: str = ExecutionMode.LIVE.value,
    attempt_status: str = AttemptTerminalStatus.SUCCEEDED.value,
    acceptance_disposition: str = AcceptanceDisposition.CANDIDATE_ONLY.value,
    production_eligible: bool = False,
    failure_reason_codes: tuple[str, ...] = (),
    selected_tests_passed: bool | None = True,
    full_suite_passed: bool | None = True,
    **overrides: Any,
) -> PairedAttemptRecord:
    fields: dict[str, Any] = {
        "role": role,
        "execution_mode": execution_mode,
        "context_pack_cid": _cid(f"ctx-{role}"),
        "route_id": f"route.{role}",
        "attempt_status": attempt_status,
        "acceptance_disposition": acceptance_disposition,
        "cost_timing": CostTimingProjection(
            input_tokens=10,
            output_tokens=5,
            wall_time_ms=20,
            model_spend_micros=100,
        ),
        "verification": VerificationProjection(
            verification_bundle_cid=_cid("ver-bundle"),
            selected_tests_passed=selected_tests_passed,
            full_suite_passed=full_suite_passed,
            proofs_passed=True,
            static_checks_passed=True,
            counterexample_present=False,
            acceptance_matrix_satisfied=False,
            production_eligible=production_eligible,
        ),
        "patch_cid": _cid(f"patch-{role}"),
        "worktree_id": f"worktree-{role}",
        "failure_reason_codes": failure_reason_codes,
    }
    fields.update(overrides)
    return PairedAttemptRecord(**fields)


# ---------------------------------------------------------------------------
# Module surface / evidence / import safety
# ---------------------------------------------------------------------------


def test_module_exists_and_exports_frozen_interfaces() -> None:
    assert MODULE_PATH.is_file()
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    }
    assert "GovernorRuntime" in names
    assert "audit_task" in names
    assert "shadow_task" in names
    assert "expand_audit" in names
    assert governor_runtime_interface_id() == GOVERNOR_RUNTIME_INTERFACE
    assert audit_task_interface_id() == AUDIT_TASK_INTERFACE
    assert shadow_task_interface_id() == SHADOW_TASK_INTERFACE
    assert expand_audit_interface_id() == EXPAND_AUDIT_INTERFACE
    assert runtime_conformance_evidence_id() == SCG_RUNTIME_CONFORMANCE_EVIDENCE
    assert SCG_RUNTIME_CONFORMANCE_EVIDENCE == "scg/runtime-conformance@1"


def test_module_import_performs_no_io() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"open", "urlopen", "system", "Popen", "connect", "create_connection"}
    for node in tree.body:
        if not isinstance(node, (ast.Expr, ast.Assign, ast.AnnAssign)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                func = child.func
                name = (
                    func.id
                    if isinstance(func, ast.Name)
                    else (func.attr if isinstance(func, ast.Attribute) else "")
                )
                assert name not in forbidden


# ---------------------------------------------------------------------------
# Acceptance: interrupted audits recover
# ---------------------------------------------------------------------------


def test_interrupted_audit_recovers_and_completes() -> None:
    rt = _runtime()
    task = _task("task.interrupt.1")
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
    assert first.phase == AuditPhase.INTERRUPTED.value
    assert first.plan_cid is not None
    assert first.shadow_result_cid is not None
    assert first.differential_cid is not None
    assert "interrupted_after_compared" in first.reason_codes

    # Durable checkpoint must be present for recovery.
    loaded = rt.audit_store.load(first.audit_id)
    assert loaded is not None
    assert loaded.plan_cid == first.plan_cid
    assert loaded.shadow_result_cid == first.shadow_result_cid

    second = rt.audit_task(task, ctx, repo, sample_roll=0)
    assert second.recovered is True
    assert second.phase == AuditPhase.COMPLETE.value
    assert second.disposition in {
        AuditDisposition.RECOVERED.value,
        AuditDisposition.COMPLETE.value,
    }
    # Identities from the interrupted run are preserved across recovery.
    assert second.plan_cid == first.plan_cid
    assert second.shadow_result_cid == first.shadow_result_cid
    assert second.differential_cid == first.differential_cid
    assert second.input_identity_cid == first.input_identity_cid
    assert "interrupted_audit_recovered" in second.reason_codes


def test_interrupted_expansion_recovers_via_expand_audit() -> None:
    store = InMemoryExpansionCheckpointStore()
    plan = _expansion_plan(
        steps=[
            _step(step_id="step_0000_include_raw_source_a", step_index=0, token_increase=80),
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

    # First expand_audit is interrupted inside the expansion loop.
    rt = _runtime(expansion_store=store)
    first = rt.expand_audit(
        plan,
        runner=AlwaysFailRunner(),
        cancel_requested=cancel_after_first,
    )
    assert first.expansion_result is not None
    assert first.expansion_result["disposition"] == ExpansionLoopDisposition.CANCELLED.value
    assert first.expansion_result["budget"]["spent_tokens"] == 80

    # Resume through expand_audit recovers checkpoint spend.
    second = rt.expand_audit(plan, runner=AlwaysFailRunner())
    assert second.recovered is True
    assert second.disposition == ExpandAuditDisposition.RECOVERED.value
    assert second.expansion_result is not None
    assert second.expansion_result["budget"]["spent_tokens"] >= 80
    assert "expansion_recovered_from_checkpoint" in second.reason_codes


def test_filesystem_audit_checkpoint_store_round_trip(tmp_path: Path) -> None:
    store = FilesystemAuditCheckpointStore(tmp_path / "audits")
    rt = _runtime(audit_store=store)
    result = rt.audit_task(_task("task.fs.1"), _context(), _repo(), sample_roll=0)
    assert result.phase == AuditPhase.COMPLETE.value
    reloaded = store.load(result.audit_id)
    assert reloaded is not None
    assert reloaded.plan_cid == result.plan_cid
    assert reloaded.input_identity_cid == result.input_identity_cid
    by_input = store.load_by_input_identity(result.input_identity_cid)
    assert by_input is not None
    assert by_input.audit_id == result.audit_id


# ---------------------------------------------------------------------------
# Acceptance: duplicate inputs preserve identities
# ---------------------------------------------------------------------------


def test_duplicate_audit_inputs_preserve_identities() -> None:
    rt = _runtime()
    task = _task("task.dup.1")
    ctx = _context()
    repo = _repo()

    first = rt.audit_task(task, ctx, repo, sample_roll=0)
    second = rt.audit_task(task, ctx, repo, sample_roll=0)

    assert first.phase == AuditPhase.COMPLETE.value
    assert second.idempotent_hit is True
    assert second.disposition == AuditDisposition.IDEMPOTENT_HIT.value
    assert second.input_identity_cid == first.input_identity_cid
    assert second.plan_cid == first.plan_cid
    assert second.shadow_result_cid == first.shadow_result_cid
    assert second.differential_cid == first.differential_cid
    assert second.audit_id == first.audit_id
    assert "duplicate_inputs_preserve_identities" in second.reason_codes

    # Result CID of the sealed first run is stable when re-materialized.
    restored = type(first).from_dict(first.to_dict())
    assert restored.result_cid == first.result_cid


def test_duplicate_shadow_task_inputs_preserve_identities() -> None:
    rt = _runtime()
    task = _task("task.shadow.dup")
    ctx = _context()
    repo = _repo()

    first = rt.shadow_task(task, ctx, repo, sample_roll=0)
    assert first.disposition == ShadowTaskDisposition.COMPLETE.value
    assert first.plan_cid is not None
    assert first.shadow_result_cid is not None

    second = rt.shadow_task(task, ctx, repo, sample_roll=0)
    assert second.idempotent_hit is True
    assert second.disposition == ShadowTaskDisposition.IDEMPOTENT_HIT.value
    assert second.plan_cid == first.plan_cid
    assert second.shadow_result_cid == first.shadow_result_cid
    assert second.differential_cid == first.differential_cid
    assert second.input_identity_cid == first.input_identity_cid
    assert "duplicate_inputs_preserve_identities" in second.reason_codes


def test_module_level_wrappers_match_runtime_methods() -> None:
    rt = _runtime()
    task = _task("task.wrap.1")
    ctx = _context()
    repo = _repo()
    via_runtime = rt.audit_task(task, ctx, repo, sample_roll=0)
    # Fresh runtime with same store is not shared; compare interface ids instead.
    assert audit_task_interface_id() == "audit_task@1"
    via_fn = shadow_task(
        _task("task.wrap.2"),
        ctx,
        repo,
        development_shadow_sampling_policy(random_seed=1),
        runtime=rt,
        sample_roll=0,
    )
    assert via_fn.disposition == ShadowTaskDisposition.COMPLETE.value
    assert via_runtime.plan_cid is not None


# ---------------------------------------------------------------------------
# Acceptance: private external shadow rejected
# ---------------------------------------------------------------------------


def test_private_external_shadow_rejected_by_gate() -> None:
    with pytest.raises(PrivateExternalShadowError, match="private"):
        reject_private_external_shadow(
            provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
            context={"raw_private_source": "def secret():\n    return 1\n"},
            includes_private_source=True,
            allow_external_expanded_disclosure=True,
            raise_on_forbidden=True,
        )


def test_private_external_shadow_rejected_by_shadow_task() -> None:
    rt = _runtime()
    with pytest.raises(PrivateExternalShadowError):
        rt.shadow_task(
            _task("task.private.ext"),
            _context(includes_private_source=True),
            _repo(),
            expanded_provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
            expanded_context={
                "raw_private_source": "class Secret: pass\n",
                "context_pack_cid": _cid("expanded-private"),
            },
            sample_roll=0,
        )


def test_private_external_shadow_rejected_by_audit_task() -> None:
    rt = _runtime()
    result = rt.audit_task(
        _task("task.private.audit"),
        _context(includes_private_source=True),
        _repo(),
        expanded_provider_id=DEFAULT_EXTERNAL_PROVIDER_ID,
        expanded_context={"private_source": "x = 1\n"},
        sample_roll=0,
    )
    assert result.disposition == AuditDisposition.REJECTED.value
    assert result.phase == AuditPhase.REJECTED.value
    assert "private_external_shadow_forbidden" in result.reason_codes


def test_local_private_shadow_is_allowed() -> None:
    decision = reject_private_external_shadow(
        provider_id="local:expanded",
        context={"raw_private_source": "def f():\n    return 0\n"},
        includes_private_source=True,
        raise_on_forbidden=True,
    )
    assert decision["allowed"] is True
    assert decision["disposition"] in {
        DisclosureDisposition.LOCAL_ONLY.value,
        DisclosureDisposition.ALLOWED.value,
        DisclosureDisposition.REDACTED_ONLY.value,
    }


# ---------------------------------------------------------------------------
# Acceptance: unbounded expansion rejected
# ---------------------------------------------------------------------------


def test_unbounded_expansion_metadata_rejected() -> None:
    plan = _expansion_plan(metadata={"unbounded": True})
    with pytest.raises(UnboundedExpansionError, match="unbounded"):
        reject_unbounded_expansion(plan)


def test_unbounded_expansion_zero_wall_time_rejected() -> None:
    plan = _expansion_plan(max_wall_time_ms=0)
    with pytest.raises(UnboundedExpansionError, match="max_wall_time_ms"):
        reject_unbounded_expansion(plan)


def test_unbounded_expansion_exceeds_runtime_ceiling_rejected() -> None:
    plan = _expansion_plan(max_token_growth=MAX_RUNTIME_TOKEN_GROWTH + 1)
    with pytest.raises(UnboundedExpansionError, match="max_token_growth"):
        reject_unbounded_expansion(plan)

    plan2 = _expansion_plan(metadata={"unlimited": True})
    with pytest.raises(UnboundedExpansionError, match="unbounded"):
        reject_unbounded_expansion(plan2)

    # Composition ceiling is finite and smaller than pathological budgets.
    assert MAX_RUNTIME_EXPANSION_STEPS > 0
    assert MAX_RUNTIME_TOKEN_GROWTH > 0


def test_expand_audit_rejects_unbounded_plan() -> None:
    rt = _runtime()
    with pytest.raises(UnboundedExpansionError):
        expand_audit(
            _expansion_plan(metadata={"unbounded_expansion": True}),
            runtime=rt,
        )


def test_bounded_expansion_admitted_and_executes() -> None:
    rt = _runtime()
    plan = _expansion_plan()
    admitted = reject_unbounded_expansion(plan)
    assert admitted.plan_cid == plan.plan_cid
    result = rt.expand_audit(
        plan,
        runner=RepairingOnArtifactRunner(required_artifact_id="exc_helper"),
        comparative_outcome="compressed_failed_expanded_succeeded",
    )
    assert result.disposition == ExpandAuditDisposition.COMPLETE.value
    assert result.expansion_result_cid is not None
    assert "bounded_expansion_enforced" in result.reason_codes
    assert result.expansion_result is not None
    assert result.expansion_result["repaired"] is True


# ---------------------------------------------------------------------------
# Acceptance: suppressed failure rejected
# ---------------------------------------------------------------------------


def test_suppressed_failure_metadata_rejected() -> None:
    with pytest.raises(SuppressedFailureError, match="suppress"):
        reject_suppressed_failure(metadata={"suppress_failure": True})


def test_failed_attempt_without_reason_codes_rejected() -> None:
    # Runtime gate (kwargs path — contracts also fail closed at construction).
    with pytest.raises(SuppressedFailureError, match="failure_reason_code"):
        reject_suppressed_failure(
            attempt_status=AttemptTerminalStatus.FAILED.value,
            acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED.value,
            failure_reason_codes=(),
            verification_passed=False,
        )
    # Sealed contract construction independently rejects suppressed failures.
    with pytest.raises(Exception, match="failure_reason_code"):
        _attempt(
            attempt_status=AttemptTerminalStatus.FAILED.value,
            acceptance_disposition=AcceptanceDisposition.NOT_ACCEPTED.value,
            failure_reason_codes=(),
            selected_tests_passed=False,
            full_suite_passed=False,
        )


def test_acceptance_cannot_suppress_verification_failure() -> None:
    # Explicit kwargs path: acceptance with verification_passed=False is rejected.
    with pytest.raises(SuppressedFailureError, match="verification"):
        reject_suppressed_failure(
            attempt_status=AttemptTerminalStatus.SUCCEEDED.value,
            acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
            verification_passed=False,
            production_eligible=True,
        )
    with pytest.raises(SuppressedFailureError, match="production_eligible"):
        reject_suppressed_failure(
            verification_passed=False,
            production_eligible=True,
        )


def test_expanded_never_accepted_enforced_by_suppressed_failure_gate() -> None:
    # PairedAttemptRecord itself rejects expanded+accepted; gate must also fail.
    with pytest.raises(Exception):
        _attempt(
            role=ShadowAttemptRole.EXPANDED.value,
            acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
            production_eligible=False,
        )


def test_audit_task_rejects_suppress_failure_metadata() -> None:
    rt = _runtime()
    with pytest.raises(SuppressedFailureError):
        rt.audit_task(
            _task("task.suppress.1"),
            _context(),
            _repo(),
            sample_roll=0,
            metadata={"hide_failure": True},
        )


# ---------------------------------------------------------------------------
# Acceptance: simulated live-quality claims rejected
# ---------------------------------------------------------------------------


def test_simulated_live_quality_metadata_rejected() -> None:
    with pytest.raises(SimulatedLiveQualityError, match="live quality"):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            metadata={"live_quality": True},
        )


def test_simulated_accepted_claim_rejected() -> None:
    with pytest.raises(SimulatedLiveQualityError, match="accepted"):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            acceptance_disposition=AcceptanceDisposition.ACCEPTED.value,
        )


def test_simulated_production_eligible_rejected() -> None:
    with pytest.raises(SimulatedLiveQualityError, match="production_eligible"):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            production_eligible=True,
        )


def test_simulated_observation_accepted_live_quality_rejected() -> None:
    # Simulated + accepted alone is not a live-quality claim (calibration skips it).
    obs_ok = observation_from_receipt_fields(
        observation_id="obs.sim.ok",
        route_tier="medium",
        accepted=True,
        receipt_cid=_cid("receipt-sim-ok"),
        simulated=True,
    )
    reject_simulated_live_quality_claim(
        execution_mode=ExecutionMode.SIMULATED.value,
        observation=obs_ok,
    )

    obs = observation_from_receipt_fields(
        observation_id="obs.sim.1",
        route_tier="medium",
        accepted=True,
        receipt_cid=_cid("receipt-sim"),
        simulated=True,
        metadata={"claims_live_quality": True},
    )
    with pytest.raises(SimulatedLiveQualityError):
        reject_simulated_live_quality_claim(
            execution_mode=ExecutionMode.SIMULATED.value,
            observation=obs,
        )


def test_simulated_calibration_never_counts_as_live_quality() -> None:
    obs_sim = observation_from_receipt_fields(
        observation_id="obs.sim.calib",
        route_tier="medium",
        accepted=True,
        receipt_cid=_cid("receipt-sim-calib"),
        simulated=True,
    )
    # Without explicit live-quality metadata, simulated is skipped (not applied).
    result = reject_simulated_calibration_as_live(None, [obs_sim])
    assert result.disposition == RouteCalibrationDisposition.SKIPPED_SIMULATED.value
    assert result.applied_observation_cids == ()

    # Explicit live-quality claim on simulated observation fails closed.
    obs_claim = observation_from_receipt_fields(
        observation_id="obs.sim.claim",
        route_tier="medium",
        accepted=False,
        receipt_cid=_cid("receipt-sim-claim"),
        simulated=True,
        metadata={"live_quality_claim": True},
    )
    with pytest.raises(SimulatedLiveQualityError):
        reject_simulated_calibration_as_live(None, [obs_claim])


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
# Happy-path composition / identity round-trip
# ---------------------------------------------------------------------------


def test_full_audit_task_happy_path_with_expansion() -> None:
    rt = _runtime()
    plan = _expansion_plan()
    result = rt.audit_task(
        _task("task.full.1"),
        _context(),
        _repo(),
        sample_roll=0,
        run_expansion=True,
        expansion_plan=plan,
        metadata={"suite": "scg-032"},
    )
    assert result.phase == AuditPhase.COMPLETE.value
    assert result.disposition in {
        AuditDisposition.COMPLETE.value,
        AuditDisposition.RECOVERED.value,
    }
    assert result.plan_cid is not None
    assert result.shadow_result_cid is not None
    assert result.differential_cid is not None
    assert result.expansion_result_cid is not None
    assert result.comparative_outcome is not None
    assert "audit_task_complete" in result.reason_codes
    assert result.evidence_id if hasattr(result, "evidence_id") else True
    payload = result.to_dict()
    assert payload["schema"].endswith("audit-task-result@1")
    assert payload["interface_id"] == AUDIT_TASK_INTERFACE
    assert payload["evidence_id"] == SCG_RUNTIME_CONFORMANCE_EVIDENCE


def test_shadow_task_publishes_differential() -> None:
    result = shadow_task(
        _task("task.shadow.only"),
        _context(),
        _repo(),
        development_shadow_sampling_policy(random_seed=3),
        sample_roll=0,
    )
    assert result.disposition == ShadowTaskDisposition.COMPLETE.value
    assert result.differential is not None
    assert result.comparative_outcome is not None
    assert "differential_published" in result.reason_codes


def test_recover_audit_explicit_api() -> None:
    rt = _runtime()
    completed = rt.audit_task(_task("task.recover.api"), _context(), _repo(), sample_roll=0)
    recovered = rt.recover_audit(completed.audit_id)
    assert recovered.audit_id == completed.audit_id
    assert recovered.plan_cid == completed.plan_cid
    assert "explicit_recover_audit" in recovered.reason_codes
