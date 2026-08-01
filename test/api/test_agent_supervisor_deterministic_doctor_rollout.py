"""LPR-041: report-only through narrow-auto rollout controls and operator validation.

Acceptance:

* immutable bounded config defaults to report-only with deterministic
  narrow-auto false, remote embeddings/network/LLM/remote model-provider calls
  false, and exact-root/proof-cache-revalidation/native-reconstruction/
  all-callers/sandbox/lease/atomic/fixed-point gates true;
* limits for findings/candidates/queries/operators/plan steps/iterations/
  files/bytes/processes/time/CPU/memory are defined;
* promotion is manual and monotonic;
* any nonzero safety floor, root/schema/capability drift, embedding canary
  failure, reconstruction/isolation loss, transaction/rollback failure or
  material resource regression rolls back or disables auto;
* ordinary lifecycle doctor remains read-only/idempotent;
* optional provider absence is actionable but does not block report-only
  startup.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.control import deterministic_doctor_rollout as rollout

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = (
    _REPO_ROOT / "scripts" / "ops" / "agent_supervisor" / "validate_deterministic_doctor.py"
)
_GUIDE_PATH = _REPO_ROOT / "docs" / "guides" / "DETERMINISTIC_DOCTOR_GUIDE.md"
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "control"
    / "deterministic_doctor_rollout.py"
)
_CONFIG_PATH = _REPO_ROOT / "config" / "agent_supervisor_deterministic_doctor.json"


def _load_cli():
    name = "validate_deterministic_doctor_lpr041"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


# ---------------------------------------------------------------------------
# Deliverables
# ---------------------------------------------------------------------------


def test_declared_outputs_exist() -> None:
    assert _MODULE_PATH.is_file()
    assert _CONFIG_PATH.is_file()
    assert _SCRIPT_PATH.is_file()
    assert _GUIDE_PATH.is_file()
    assert Path(__file__).is_file()


def test_interfaces_and_schemas_are_stable() -> None:
    assert rollout.ROLLOUT_POLICY_INTERFACE == "DeterministicDoctorRolloutPolicy@1"
    assert rollout.ROLLOUT_DECISION_INTERFACE == "DeterministicDoctorRolloutDecision@1"
    assert rollout.ROLLBACK_GATE_INTERFACE == "DeterministicDoctorRollbackGate@1"
    assert rollout.VALIDATOR_INTERFACE == "DeterministicDoctorOperationsValidator@1"
    assert rollout.METRICS_INTERFACE == "DeterministicDoctorMetrics@1"
    assert rollout.SERVICE_INTERFACE == "DeterministicDoctorService@1"
    assert rollout.SUPERVISOR_CONTROL_SERVICE_INTERFACE == "SupervisorControlService@1"
    assert rollout.TASK_ID == "LPR-041"
    assert rollout.GOAL_ID == "LPR-G110"
    assert hasattr(rollout, "DeterministicDoctorMode")
    assert hasattr(rollout, "DeterministicDoctorRolloutPolicy")
    assert hasattr(rollout, "DeterministicDoctorRolloutDecision")
    assert hasattr(rollout, "DeterministicDoctorRollbackGate")
    assert hasattr(rollout, "DeterministicDoctorOperationsValidator")


# ---------------------------------------------------------------------------
# Defaults / config
# ---------------------------------------------------------------------------


def test_report_only_is_default_mode() -> None:
    policy = rollout.default_rollout_policy()
    assert policy.mode_value == "report_only"
    assert policy.mode is rollout.DeterministicDoctorMode.REPORT_ONLY
    assert policy.mutation_authorized is False
    assert policy.completion_authoritative is False
    assert policy.enabled is False
    assert policy.allow_plan is False
    assert policy.allow_sandbox_auto is False
    assert policy.allow_narrow_auto is False
    assert policy.narrow_autonomous_mutation_enabled is False
    assert policy.kill_switch_engaged is False
    assert set(rollout.ROLLOUT_STAGES) == {
        "report_only",
        "plan",
        "sandbox_auto",
        "narrow_auto",
    }


def test_hard_false_flags_and_hard_true_gates() -> None:
    policy = rollout.default_rollout_policy()
    flags = policy.feature_flags()
    for name in (
        "llm_router_enabled",
        "llm_invocations_allowed",
        "remote_model_provider_calls_allowed",
        "remote_embeddings_allowed",
        "network_access_allowed",
        "target_code_import_allowed",
    ):
        assert flags[name] is False
    for name, value in policy.gates().items():
        assert value is True, name
    for name, value in policy.semantic_authority_flags().items():
        assert value is False, name
    # Exact-root / proof-cache / native reconstruction / all-callers / sandbox /
    # lease / atomic / fixed-point gates
    assert policy.exact_evidence_snapshot_required is True
    assert policy.proof_cache_binding_revalidation_required is True
    assert policy.native_kernel_reconstruction_required is True
    assert policy.complete_impact_closure_required is True
    assert policy.enforced_sandbox_required_for_target_execution is True
    assert policy.writer_lease_and_checkpoint_required is True
    assert policy.atomic_scc_transaction_required is True
    assert policy.logic_and_program_fixed_point_required is True


def test_limits_cover_required_resource_dimensions() -> None:
    policy = rollout.default_rollout_policy()
    required = {
        "max_findings",
        "max_candidates_per_finding",
        "max_graph_nodes_per_query",
        "max_operators_per_finding",
        "max_plan_steps",
        "max_fixed_point_iterations",
        "max_changed_files",
        "max_changed_bytes",
        "max_processes",
        "max_wall_time_seconds",
        "max_cpu_time_seconds",
        "max_memory_bytes",
    }
    assert required <= set(policy.limits)
    assert set(policy.limits) == set(rollout.LIMIT_KEYS)
    for key, value in policy.limits.items():
        assert isinstance(value, int) and value > 0, key
    result = rollout.check_limits()
    assert result.status is rollout.CheckStatus.PASS


def test_config_file_matches_immutable_defaults() -> None:
    payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
    assert payload["schema"] == rollout.CONFIG_SCHEMA
    assert payload["default_mode"] == "report_only"
    assert payload["feature_flags"]["narrow_autonomous_mutation_enabled"] is False
    for name in (
        "remote_embeddings_allowed",
        "network_access_allowed",
        "llm_invocations_allowed",
        "remote_model_provider_calls_allowed",
    ):
        assert payload["feature_flags"][name] is False
    for name in rollout.HARD_TRUE_GATES:
        assert payload["gates"][name] is True
    assert payload["promotion"]["manual"] is True
    assert payload["promotion"]["monotonic"] is True
    assert payload["lifecycle_doctor"]["read_only"] is True
    assert payload["lifecycle_doctor"]["idempotent"] is True
    assert payload["optional_providers"]["absence_blocks_report_only_startup"] is False
    result = rollout.check_config_defaults(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["config_identity"].startswith("sha256:")
    policy = rollout.DeterministicDoctorRolloutPolicy.from_config_mapping(payload)
    assert policy.mode_value == "report_only"
    assert policy.floors_hold()


def test_cannot_enable_model_or_network_flags() -> None:
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="must be false"):
        rollout.DeterministicDoctorRolloutPolicy(llm_invocations_allowed=True)
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="must be false"):
        rollout.DeterministicDoctorRolloutPolicy(remote_embeddings_allowed=True)
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="must be false"):
        rollout.DeterministicDoctorRolloutPolicy(network_access_allowed=True)
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="must be false"):
        rollout.DeterministicDoctorRolloutPolicy(remote_model_provider_calls_allowed=True)
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="must remain enabled"):
        rollout.DeterministicDoctorRolloutPolicy(atomic_scc_transaction_required=False)
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="completion authority"):
        rollout.DeterministicDoctorRolloutPolicy(completion_authoritative=True)


# ---------------------------------------------------------------------------
# Promotion
# ---------------------------------------------------------------------------


def test_plan_sandbox_narrow_require_explicit_policy() -> None:
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="explicit scoped"):
        rollout.DeterministicDoctorRolloutPolicy(mode=rollout.DeterministicDoctorMode.PLAN)
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="explicit scoped"):
        rollout.DeterministicDoctorRolloutPolicy(
            mode=rollout.DeterministicDoctorMode.SANDBOX_AUTO
        )
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="explicit scoped"):
        rollout.DeterministicDoctorRolloutPolicy(
            mode=rollout.DeterministicDoctorMode.NARROW_AUTO
        )
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="explicit scoped"):
        rollout.DeterministicDoctorRolloutPolicy(
            mode=rollout.DeterministicDoctorMode.PLAN,
            explicit_policy_document="",
            repository_id="repository:x",
        )

    plan = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.PLAN,
        explicit_policy_document="policy://reviewed/plan",
        repository_id="repository:demo",
    )
    assert plan.mode_value == "plan"
    assert plan.has_explicit_scoped_policy() is True
    assert plan.mutation_authorized is False

    sandbox = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.SANDBOX_AUTO,
        explicit_policy_document="policy://reviewed/sandbox",
        repository_id="repository:demo",
    )
    assert sandbox.mode_value == "sandbox_auto"
    assert sandbox.mutation_authorized is False

    narrow = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    assert narrow.mode_value == "narrow_auto"
    assert narrow.mutation_authorized is True
    assert narrow.narrow_autonomous_mutation_enabled is True


def test_promotion_is_manual_and_monotonic() -> None:
    result = rollout.check_promotion_monotonicity()
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["ranks"] == [0, 1, 2, 3]
    assert result.evidence["promotion_manual"] is True
    assert result.evidence["promotion_monotonic"] is True


def test_kill_switch_blocks_elevation_and_forces_report_only() -> None:
    with pytest.raises(rollout.DeterministicDoctorRolloutError, match="kill switch"):
        rollout.elevate_rollout_policy(
            mode=rollout.DeterministicDoctorMode.PLAN,
            explicit_policy_document="policy://reviewed/plan",
            repository_id="repository:demo",
            kill_switch_engaged=True,
        )
    narrow = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    killed = rollout.engage_kill_switch(narrow)
    assert killed.mode_value == "report_only"
    assert killed.kill_switch_engaged is True
    assert killed.mutation_authorized is False
    assert killed.narrow_autonomous_mutation_enabled is False
    assert killed.enabled is False


def test_narrow_auto_requires_all_gates() -> None:
    narrow = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    base = dict(
        unique_target=True,
        reconstructed=True,
        complete_frontier=True,
        sandbox_isolated=True,
        lease_held=True,
        atomic_transaction=True,
        fixed_point_ready=True,
    )
    assert narrow.allows_automated_mutation(**base)
    assert not narrow.allows_automated_mutation(**{**base, "complete_frontier": False})
    assert not narrow.allows_automated_mutation(**{**base, "unique_target": False})
    assert not narrow.allows_automated_mutation(**{**base, "reconstructed": False})
    assert not narrow.allows_automated_mutation(**{**base, "sandbox_isolated": False})
    assert not narrow.allows_automated_mutation(**{**base, "lease_held": False})
    assert not narrow.allows_automated_mutation(**{**base, "atomic_transaction": False})
    assert not narrow.allows_automated_mutation(**{**base, "fixed_point_ready": False})
    assert not narrow.allows_automated_mutation(
        **{**base, "approval_class": "public_api_or_schema"}
    )
    report_only = rollout.default_rollout_policy()
    assert not report_only.allows_automated_mutation(**base)


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_triggers_demote_and_disable_auto() -> None:
    policy = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    floor = rollout.evaluate_rollback(
        policy,
        safety_floors={
            **{k: 0 for k in rollout.SAFETY_FLOOR_KEYS},
            "missed_mandatory_caller_rate": 1,
        },
    )
    assert floor is not None
    assert floor.reason is rollout.RollbackReason.NONZERO_SAFETY_FLOOR
    demoted = rollout.apply_rollback(policy, floor)
    assert demoted.mutation_authorized is False
    assert demoted.narrow_autonomous_mutation_enabled is False
    assert demoted.mode_value == "report_only"

    for kwargs, reason in (
        (
            dict(root_schema_capability_drift=True),
            rollout.RollbackReason.ROOT_SCHEMA_CAPABILITY_DRIFT,
        ),
        (
            dict(embedding_canary_failure=True),
            rollout.RollbackReason.EMBEDDING_CANARY_FAILURE,
        ),
        (
            dict(reconstruction_isolation_loss=True),
            rollout.RollbackReason.RECONSTRUCTION_ISOLATION_LOSS,
        ),
        (
            dict(transaction_rollback_failure=True),
            rollout.RollbackReason.TRANSACTION_ROLLBACK_FAILURE,
        ),
        (
            dict(material_resource_regression=True),
            rollout.RollbackReason.MATERIAL_RESOURCE_REGRESSION,
        ),
        (dict(kill_switch=True), rollout.RollbackReason.KILL_SWITCH),
        (
            dict(capability_regression=("retrieval",)),
            rollout.RollbackReason.CAPABILITY_REGRESSION,
        ),
    ):
        receipt = rollout.evaluate_rollback(policy, **kwargs)
        assert receipt is not None, kwargs
        assert receipt.reason is reason
        applied = rollout.apply_rollback(policy, receipt)
        assert applied.mutation_authorized is False
        assert applied.narrow_autonomous_mutation_enabled is False

    assert (
        rollout.evaluate_rollback(
            policy, safety_floors={k: 0 for k in rollout.SAFETY_FLOOR_KEYS}
        )
        is None
    )


def test_rollback_gate_class_and_check_pass() -> None:
    gate = rollout.DeterministicDoctorRollbackGate()
    assert gate.INTERFACE == "DeterministicDoctorRollbackGate@1"
    assert gate.evaluate(safety_floors={k: 0 for k in rollout.SAFETY_FLOOR_KEYS}) is None
    result = rollout.check_rollback_gates()
    assert result.status is rollout.CheckStatus.PASS, result.detail


def test_rollout_decision_disables_auto_on_regression() -> None:
    narrow = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    healthy = rollout.evaluate_rollout_decision(narrow)
    assert healthy.effective_mode_value == "narrow_auto"
    assert healthy.mutation_authorized is True
    assert healthy.narrow_auto_disabled is False

    drifted = rollout.evaluate_rollout_decision(
        narrow, root_schema_capability_drift=True
    )
    assert drifted.effective_mode_value == "report_only"
    assert drifted.mutation_authorized is False
    assert drifted.narrow_auto_disabled is True
    assert "root_schema_capability_drift" in drifted.reason_codes

    breached_policy = rollout.elevate_rollout_policy(
        mode=rollout.DeterministicDoctorMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
        safety_floors={
            **{k: 0 for k in rollout.SAFETY_FLOOR_KEYS},
            "embedding_canary_failure_rate": 1,
        },
    )
    decision = rollout.evaluate_rollout_decision(breached_policy)
    assert decision.effective_mode_value == "report_only"
    assert decision.floor_breaches == ("embedding_canary_failure_rate",)


# ---------------------------------------------------------------------------
# Lifecycle / providers / ops
# ---------------------------------------------------------------------------


def test_lifecycle_doctor_is_read_only_and_idempotent() -> None:
    result = rollout.check_lifecycle_doctor_readonly(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["read_only"] is True
    assert result.evidence["idempotent"] is True
    policy = rollout.default_rollout_policy()
    a = rollout.evaluate_rollout_decision(policy)
    b = rollout.evaluate_rollout_decision(policy)
    assert a.decision_id == b.decision_id
    assert a.mutation_authorized is False
    assert a.effective_mode_value == "report_only"
    doctor_a = rollout.doctor(_REPO_ROOT)
    doctor_b = rollout.doctor(_REPO_ROOT)
    assert doctor_a["report_id"] == doctor_b["report_id"]
    assert doctor_a["mutation_authorized"] is False
    assert doctor_a["completion_authoritative"] is False


def test_optional_provider_absence_does_not_block_report_only() -> None:
    result = rollout.check_optional_provider_absence()
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["absence_blocks_report_only_startup"] is False
    assert result.evidence["absence_is_actionable"] is True
    assert result.evidence["report_only_startup_ok"] is True
    # Startup path succeeds even if optional modules are missing.
    assert isinstance(result.evidence["missing_modules"], list)


def test_feature_flag_check_passes() -> None:
    result = rollout.check_feature_flags()
    assert result.status is rollout.CheckStatus.PASS


def test_run_all_checks() -> None:
    report = rollout.run_all_checks(_REPO_ROOT)
    assert report["schema"] == rollout.VALIDATOR_SCHEMA
    assert report["default_mode"] == "report_only"
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False
    assert report["valid"] is True, report.get("failed")
    names = {item["name"] for item in report["checks"]}
    assert {
        "declared_artifacts",
        "config_defaults",
        "feature_flags",
        "resource_limits",
        "promotion_monotonicity",
        "rollback_gates",
        "lifecycle_doctor_readonly",
        "optional_provider_absence",
        "guide_boundaries",
        "related_surfaces",
    } <= names
    assert report["config_identity"].startswith("sha256:")
    assert report["report_id"].startswith("sha256:")


def test_doctor_and_status_commands() -> None:
    doctor = rollout.doctor(_REPO_ROOT)
    assert doctor["command"] == "doctor"
    assert doctor["default_mode"] == "report_only"
    assert doctor["valid"] is True
    st = rollout.status(_REPO_ROOT)
    assert st["command"] == "status"
    assert st["mode"] == "report_only"
    assert st["effective_mode"] == "report_only"
    assert st["valid"] is True
    assert st["kill_switch_engaged"] is False
    assert st["lifecycle_doctor"]["status"] == "pass"
    assert st["optional_providers"]["status"] == "pass"


def test_operations_validator_facade() -> None:
    ops = rollout.DeterministicDoctorOperationsValidator(_REPO_ROOT)
    report = ops.run_all()
    assert report["valid"] is True
    assert ops.status()["mode"] == "report_only"
    assert ops.doctor()["command"] == "doctor"
    payload = ops.to_dict()
    assert payload["interface"] == "DeterministicDoctorOperationsValidator@1"
    assert payload["mutation_authorized"] is False


def test_guide_documents_modes_kill_switch_floors_lifecycle_providers() -> None:
    text = _GUIDE_PATH.read_text(encoding="utf-8")
    lower = text.casefold()
    assert "report-only" in lower or "report_only" in lower
    assert "plan" in lower
    assert "sandbox" in lower
    assert "narrow-auto" in lower or "narrow_auto" in lower or "narrow auto" in lower
    assert "kill switch" in lower
    assert "rollback" in lower
    assert "safety floor" in lower
    assert "promotion" in lower
    assert "manual" in lower and "monotonic" in lower
    assert "lifecycle" in lower
    assert "optional provider" in lower
    assert "trust" in lower
    assert "does not block report-only" in lower or "does not block report only" in lower
    result = rollout.check_guide_boundaries(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_policy_and_checks(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["policy", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "report_only"
    assert payload["mutation_authorized"] is False
    assert payload["feature_flags"]["narrow_autonomous_mutation_enabled"] is False

    assert cli.main(["check-flags", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-config", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-limits", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-promotion", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-rollback", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-lifecycle", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-providers", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-guide", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-artifacts", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_cli_check_all(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--check-all", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["command"] == "check-all"
    assert report["default_mode"] == "report_only"
    assert report["valid"] is True
    assert code == 0
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["config_defaults"] == "pass"
    assert statuses["feature_flags"] == "pass"
    assert statuses["rollback_gates"] == "pass"
    assert statuses["lifecycle_doctor_readonly"] == "pass"
    assert statuses["optional_provider_absence"] == "pass"


def test_cli_status_and_doctor(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["status", "--json"]) == 0
    status_payload = json.loads(capsys.readouterr().out)
    assert status_payload["mode"] == "report_only"
    assert status_payload["valid"] is True
    assert cli.main(["doctor", "--json"]) == 0
    doctor_payload = json.loads(capsys.readouterr().out)
    assert doctor_payload["command"] == "doctor"
    assert doctor_payload["valid"] is True


def test_policy_binding_identity_is_stable() -> None:
    a = rollout.default_rollout_policy()
    b = rollout.default_rollout_policy()
    assert a.policy_binding_id == b.policy_binding_id
    assert a.policy_binding_id.startswith("sha256:")
    assert a.to_dict()["completion_authoritative"] is False
