"""LPR-020: rollout controls, operations, validation, and supervisor launch.

Acceptance:

* release validator composes protected bootstrap board/DAG doctor with exact
  two-repository bindings, import-isolation, native-execution permits,
  platform isolation, capability health, and benchmark floors;
* stages are doctor/replay, shadow default, assist, deterministic narrow-auto,
  and approval-gated behavior-complete model edit;
* independent flags disable prediction/learned ranking/Hammer/refinement/LLM/auto;
* stateful/public-schema/API/dynamic/generated/native/cross-root/new-dependency
  work remains approval-required;
* any nonzero floor, drift, reconstruction/countermodel loss, inconsistency,
  transaction, isolation, or budget regression rolls back;
* guide documents trust, safety, memory, transaction, and recovery boundaries.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import logic_repair_rollout as rollout

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "validate_tactician_hammer_logic_repair.py"
_GUIDE_PATH = _REPO_ROOT / "docs" / "guides" / "TACTICIAN_HAMMER_LOGIC_REPAIR_GUIDE.md"
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "logic_repair_rollout.py"
)
_BENCHMARK_PATH = _REPO_ROOT / "scripts" / "benchmark_tactician_hammer_logic_repair.py"


def _load_cli():
    name = "validate_tactician_hammer_logic_repair_lpr020"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


cli = _load_cli()


@pytest.fixture(scope="module")
def benchmark_report() -> dict:
    if not _BENCHMARK_PATH.is_file():
        pytest.skip("benchmark script missing")
    name = "benchmark_tactician_hammer_logic_repair_lpr019_for_lpr020"
    if name in sys.modules:
        bench = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, _BENCHMARK_PATH)
        assert spec is not None and spec.loader is not None
        bench = importlib.util.module_from_spec(spec)
        sys.modules[name] = bench
        spec.loader.exec_module(bench)
    return bench.run_benchmark()


def test_declared_outputs_exist() -> None:
    assert _MODULE_PATH.is_file()
    assert _SCRIPT_PATH.is_file()
    assert _GUIDE_PATH.is_file()
    assert Path(__file__).is_file()


def test_interfaces_and_schemas_are_stable() -> None:
    assert rollout.ROLLOUT_POLICY_INTERFACE == "LogicRepairRolloutPolicy@1"
    assert rollout.METRICS_INTERFACE == "LogicRepairMetrics@1"
    assert rollout.ROLLBACK_GATE_INTERFACE == "LogicRepairRollbackGate@1"
    assert rollout.VALIDATOR_INTERFACE == "LogicRepairOperationsValidator@1"
    assert rollout.BENCHMARK_METRICS_INTERFACE == "LogicRepairBenchmarkMetrics@1"
    assert rollout.LIVE_LOGIC_REPAIR_CONTROLLER_INTERFACE == "LiveLogicRepairController@1"
    assert rollout.PROPAGATION_COMPLETION_RECEIPT_INTERFACE == "PropagationCompletionReceipt@1"
    assert (
        rollout.LOGIC_FIXED_POINT_EVIDENCE_ATTACHMENT_INTERFACE
        == "LogicFixedPointEvidenceAttachment@1"
    )
    assert rollout.SUPERVISOR_CONTROL_SERVICE_INTERFACE == "SupervisorControlService@1"
    assert rollout.TASK_ID == "LPR-020"
    assert rollout.GOAL_ID == "LPR-G060"
    assert hasattr(rollout, "LogicRepairRolloutPolicy")
    assert hasattr(rollout, "LogicRepairMetrics")
    assert hasattr(rollout, "LogicRepairRollbackGate")
    assert hasattr(rollout, "LogicRepairOperationsValidator")


def test_shadow_is_default_mode() -> None:
    policy = rollout.default_rollout_policy()
    assert policy.mode_value == "shadow"
    assert policy.mode is rollout.RolloutMode.SHADOW
    assert policy.mutation_authorized is False
    assert policy.completion_authoritative is False
    assert policy.allow_assist is False
    assert policy.allow_narrow_auto is False
    assert policy.allow_model_edit is False
    for value in policy.feature_flags().values():
        assert value is False
    assert policy.auto_requires_unique_target is True
    assert policy.auto_requires_reconstruction is True
    assert policy.auto_requires_supported_python is True
    assert policy.auto_requires_complete_frontier is True
    assert policy.auto_requires_analytical_path is True
    assert policy.auto_requires_fixed_point is True
    assert set(rollout.ROLLOUT_STAGES) == {
        "doctor_replay", "shadow", "assist", "narrow_auto", "model_edit",
    }


def test_assist_narrow_auto_model_edit_require_explicit_policy() -> None:
    with pytest.raises(rollout.LogicRepairRolloutError, match="explicit scoped"):
        rollout.LogicRepairRolloutPolicy(mode=rollout.RolloutMode.ASSIST)
    with pytest.raises(rollout.LogicRepairRolloutError, match="explicit scoped"):
        rollout.LogicRepairRolloutPolicy(mode=rollout.RolloutMode.NARROW_AUTO)
    with pytest.raises(rollout.LogicRepairRolloutError, match="explicit scoped"):
        rollout.LogicRepairRolloutPolicy(mode=rollout.RolloutMode.MODEL_EDIT)
    with pytest.raises(rollout.LogicRepairRolloutError, match="explicit scoped"):
        rollout.LogicRepairRolloutPolicy(
            mode=rollout.RolloutMode.ASSIST,
            explicit_policy_document="",
            repository_id="repository:x",
        )

    assist = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.ASSIST,
        explicit_policy_document="policy://reviewed/assist",
        repository_id="repository:demo",
    )
    assert assist.mode_value == "assist"
    assert assist.has_explicit_scoped_policy() is True
    assert assist.mutation_authorized is False

    narrow = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    assert narrow.mode_value == "narrow_auto"
    assert narrow.mutation_authorized is True
    assert narrow.narrow_autonomous_mutation_enabled is True

    model_edit = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.MODEL_EDIT,
        explicit_policy_document="policy://reviewed/model-edit",
        repository_id="repository:demo",
    )
    assert model_edit.mode_value == "model_edit"
    assert model_edit.mutation_authorized is False


def test_narrow_auto_limited_to_complete_frontier_analytical_python() -> None:
    narrow = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    base = dict(
        transform="add_argument", unique_target=True, reconstructed=True,
        supported_python=True, complete_frontier=True, analytical_path=True,
        fixed_point_ready=True,
    )
    assert narrow.allows_automated_mutation(**base)
    assert not narrow.allows_automated_mutation(**{**base, "complete_frontier": False})
    assert not narrow.allows_automated_mutation(**{**base, "unique_target": False})
    assert not narrow.allows_automated_mutation(**{**base, "reconstructed": False})
    assert not narrow.allows_automated_mutation(**{**base, "supported_python": False})
    assert not narrow.allows_automated_mutation(**{**base, "analytical_path": False})
    assert not narrow.allows_automated_mutation(**{**base, "fixed_point_ready": False})
    shadow = rollout.default_rollout_policy()
    assert not shadow.allows_automated_mutation(**base)


def test_approval_gated_families_remain_blocked() -> None:
    narrow = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    base = dict(
        transform="add_argument", unique_target=True, reconstructed=True,
        supported_python=True, complete_frontier=True, analytical_path=True,
        fixed_point_ready=True,
    )
    assert not narrow.allows_automated_mutation(**{**base, "model_authored": True})
    assert not narrow.allows_automated_mutation(**{**base, "stateful": True})
    assert not narrow.allows_automated_mutation(**{**base, "public_schema_api": True})
    assert not narrow.allows_automated_mutation(**{**base, "dynamic": True})
    assert not narrow.allows_automated_mutation(**{**base, "generated": True})
    assert not narrow.allows_automated_mutation(**{**base, "native": True})
    assert not narrow.allows_automated_mutation(**{**base, "cross_root": True})
    assert not narrow.allows_automated_mutation(**{**base, "new_dependency": True})
    assert not narrow.allows_automated_mutation(**{**base, "behavior_complete_model_edit": True})


def test_feature_flag_check_passes() -> None:
    result = rollout.check_feature_flags()
    assert result.status is rollout.CheckStatus.PASS
    assert result.ok


def test_rollback_triggers_demote_and_revoke_mutation() -> None:
    policy = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    cap = rollout.evaluate_rollback(policy, capability_regression=("tactician", "hammer"))
    assert cap is not None
    assert cap.reason is rollout.RollbackReason.CAPABILITY_REGRESSION
    demoted = rollout.apply_rollback(policy, cap)
    assert demoted.mutation_authorized is False
    assert demoted.mode_value in {"assist", "shadow"}

    for kwargs, reason in (
        (dict(stale_roots=("code_root",)), rollout.RollbackReason.STALE_ROOT),
        (dict(open_frontier=True), rollout.RollbackReason.OPEN_FRONTIER),
        (dict(reconstruction_failed=True), rollout.RollbackReason.RECONSTRUCTION_FAILURE),
        (dict(countermodel_validation_loss=True), rollout.RollbackReason.COUNTERMODEL_VALIDATION_LOSS),
        (dict(proof_loss=True), rollout.RollbackReason.PROOF_LOSS),
        (dict(wrong_value=True), rollout.RollbackReason.WRONG_VALUE),
        (dict(missed_caller=True), rollout.RollbackReason.MISSED_CALLER),
        (dict(partial_plan=True), rollout.RollbackReason.PARTIAL_PLAN),
        (dict(false_completion=True), rollout.RollbackReason.FALSE_COMPLETION),
        (dict(isolation_regression=True), rollout.RollbackReason.ISOLATION_REGRESSION),
        (dict(budget_regression=True), rollout.RollbackReason.BUDGET_REGRESSION),
        (dict(inconsistency=True), rollout.RollbackReason.INCONSISTENCY),
        (dict(transaction_failure=True), rollout.RollbackReason.TRANSACTION_FAILURE),
    ):
        receipt = rollout.evaluate_rollback(policy, **kwargs)
        assert receipt is not None, kwargs
        assert receipt.reason is reason

    breached = rollout.LogicRepairMetrics(
        missed_caller_rate=1,
        safety_floors={**{k: 0 for k in rollout.SAFETY_FLOOR_KEYS}, "missed_resolved_caller_rate": 1},
        safety_absolute={**{k: 0 for k in rollout.SAFETY_ABSOLUTE_KEYS}, "missed_resolved_caller": 1},
    )
    metric = rollout.evaluate_rollback(policy, metrics=breached)
    assert metric is not None
    assert metric.reason is rollout.RollbackReason.METRIC_BREACH
    assert rollout.evaluate_rollback(policy, metrics=rollout.LogicRepairMetrics.empty()) is None


def test_rollback_gate_class_and_check_pass() -> None:
    gate = rollout.LogicRepairRollbackGate()
    assert gate.INTERFACE == "LogicRepairRollbackGate@1"
    assert gate.evaluate(stale_roots=()) is None
    result = rollout.check_rollback_gates()
    assert result.status is rollout.CheckStatus.PASS


def test_plan_objective_task_dag_check() -> None:
    result = rollout.check_plan_objective_task_dag(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["task_count"] >= 20
    assert "LPR-G060" in result.evidence["goal_ids"]
    assert "LPR-020" in result.evidence["task_ids"]


def test_exact_source_bindings_check() -> None:
    result = rollout.check_exact_source_bindings(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    binding = rollout.bind_exact_sources(_REPO_ROOT)
    assert binding.board_namespace == rollout.BOARD_NAMESPACE
    assert binding.task_prefix == rollout.TASK_PREFIX
    assert binding.plan_identity.startswith("sha256:")
    assert binding.fixture_manifest_identity.startswith("sha256:")
    assert binding.rollout_module_identity.startswith("sha256:")
    assert binding.binding_id.startswith("sha256:")
    assert binding.datasets_submodule == "ipfs_datasets_py"
    again = rollout.bind_exact_sources(_REPO_ROOT)
    assert again.binding_id == binding.binding_id


def test_bootstrap_board_doctor_check() -> None:
    result = rollout.check_bootstrap_board_doctor(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["lane_count"] == 4
    assert result.evidence["rollout_mode"] == "shadow"


def test_capability_health_check() -> None:
    result = rollout.check_capability_health(_REPO_ROOT, probe=True)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence.get("authoritative") is False
    assert result.evidence.get("candidate_authoritative") is False


def test_four_lane_and_launcher_checks() -> None:
    lanes = rollout.check_four_lane_sharding_and_isolation(_REPO_ROOT)
    assert lanes.status is rollout.CheckStatus.PASS, lanes.detail
    assert lanes.evidence["one_merge_queue"] is True
    assert lanes.evidence["one_refill_owner"] is True
    launcher = rollout.check_launcher_lifecycle_safety(_REPO_ROOT)
    assert launcher.status is rollout.CheckStatus.PASS, launcher.detail
    assert launcher.evidence["idempotent_start"] is True
    assert launcher.evidence["refuses_unowned_pid"] is True
    assert launcher.evidence["secrets_in_argv_or_logs"] is False


def test_proof_reconstruction_and_transaction_checks() -> None:
    proof = rollout.check_proof_reconstruction(_REPO_ROOT)
    assert proof.status is rollout.CheckStatus.PASS, proof.detail
    assert proof.evidence["auto_requires_reconstruction"] is True
    txn = rollout.check_transaction_health(_REPO_ROOT)
    assert txn.status is rollout.CheckStatus.PASS, txn.detail
    assert txn.evidence["partial_groups_cannot_merge"] is True


def test_supervisor_process_state_check(tmp_path: Path) -> None:
    result = rollout.check_supervisor_process_state(
        _REPO_ROOT, state_root=tmp_path / "missing-program"
    )
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["master_status"] == "stopped"

    program = tmp_path / "program"
    lane0 = program / "state" / "lane-0"
    lane0.mkdir(parents=True)
    (lane0 / "lpr_lane_0_supervisor_status.json").write_text(
        json.dumps({"status": "running", "pid": 2**30, "supervisor_pid": 2**30}),
        encoding="utf-8",
    )
    (lane0 / "lpr_lane_0_task_state.json").write_text(
        json.dumps({
            "status": "available", "active_task_id": "",
            "eligible_ready_count": 0, "blocked_count": 0,
        }),
        encoding="utf-8",
    )
    bad = rollout.check_supervisor_process_state(_REPO_ROOT, state_root=program, lane_count=1)
    assert bad.status is rollout.CheckStatus.FAIL


def test_benchmark_floors_check(benchmark_report: dict) -> None:
    result = rollout.check_benchmark_floors(_REPO_ROOT, run=False, report=benchmark_report)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    floors = result.evidence["safety_floors"]
    for key in rollout.SAFETY_FLOOR_KEYS:
        assert floors[key] == 0
    assert result.evidence["fixed_point_iterations_total"] >= 0
    assert set(result.evidence["benchmark_stages"]) == set(rollout.BENCHMARK_STAGES)


def test_run_all_checks_with_cached_benchmark(benchmark_report: dict) -> None:
    report = rollout.run_all_checks(
        _REPO_ROOT, run_benchmark=False, probe_capabilities=True, benchmark_report=benchmark_report,
    )
    assert report["schema"] == rollout.VALIDATOR_SCHEMA
    assert report["default_mode"] == "shadow"
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False
    assert report["valid"] is True, report.get("failed")
    names = {item["name"] for item in report["checks"]}
    assert {
        "bootstrap_board_doctor", "plan_objective_task_dag", "exact_source_bindings",
        "capability_health", "four_lane_sharding_and_isolation", "launcher_lifecycle_safety",
        "proof_reconstruction", "transaction_health", "supervisor_process_state",
        "benchmark_floors", "feature_flags", "rollback_gates", "guide_boundaries",
        "fixture_corpus_coverage",
    } <= names


def test_metrics_expose_stages_split_tokens_fixed_point(benchmark_report: dict) -> None:
    metrics = rollout.LogicRepairMetrics.from_benchmark_metrics(benchmark_report["metrics"])
    assert metrics.INTERFACE == "LogicRepairMetrics@1"
    assert metrics.floors_hold()
    assert metrics.case_count == benchmark_report["metrics"]["case_count"]
    for stage in rollout.BENCHMARK_STAGES:
        assert stage in metrics.stage_counts
    payload = metrics.to_dict()
    for key in (
        "benchmark_stages", "analytical_model_split", "tokens", "context_bytes",
        "fixed_point_iterations", "fixed_point_iterations_total", "stage_counts", "safety_floors",
    ):
        assert key in payload
    assert metrics.metrics_id.startswith("sha256:")


def test_collect_metrics_empty_without_run() -> None:
    metrics = rollout.collect_metrics(run_benchmark=False)
    assert metrics.floors_hold()
    assert metrics.case_count == 0


def test_doctor_and_status_commands(benchmark_report: dict) -> None:
    doctor = rollout.doctor(_REPO_ROOT, run_benchmark=False, probe_capabilities=True)
    full = rollout.run_all_checks(
        _REPO_ROOT, run_benchmark=False, probe_capabilities=True, benchmark_report=benchmark_report,
    )
    assert full["valid"] is True
    assert doctor["command"] == "doctor"
    assert doctor["default_mode"] == "shadow"
    st = rollout.status(_REPO_ROOT)
    assert st["command"] == "status"
    assert st["mode"] == "shadow"
    assert st["valid"] is True
    assert st["supervisor"]["evidence"]["master_status"] in {"running", "stopped"}
    assert st["four_lane_sharding_and_isolation"]["status"] == "pass"
    assert st["transaction_health"]["status"] == "pass"


def test_replay_receipt_identity_and_auto_gate() -> None:
    body = {
        "schema": "test/logic-repair-receipt@1",
        "transform": "add_argument", "unique_target": True, "reconstructed": True,
        "supported_python": True, "complete_frontier": True, "analytical_path": True,
        "fixed_point_ready": True, "language": "python",
        "roots": {"code_root": "sha256:abc", "index_root": "sha256:def"},
        "has_logic_fixed_point_attachment": True,
        "completion_interface": "PropagationCompletionReceipt@1",
    }
    sealed = {**body, "receipt_id": rollout.content_identity(body)}
    shadow_replay = rollout.replay_decision_receipt(sealed)
    assert shadow_replay["valid"] is True
    assert shadow_replay["automated_mutation_authorized"] is False

    narrow = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    auto_replay = rollout.replay_decision_receipt(sealed, policy=narrow)
    assert auto_replay["valid"] is True
    assert auto_replay["automated_mutation_authorized"] is True

    stale_replay = rollout.replay_decision_receipt(
        sealed, policy=narrow, expected_roots={"code_root": "sha256:other"},
    )
    assert stale_replay["valid"] is False
    assert stale_replay["rollback"] is not None
    assert stale_replay["rollback"]["reason"] == "stale_root"

    open_body = {**body, "complete_frontier": False, "open_frontier": True}
    open_sealed = {**open_body, "receipt_id": rollout.content_identity(open_body)}
    open_replay = rollout.replay_decision_receipt(open_sealed, policy=narrow)
    assert open_replay["automated_mutation_authorized"] is False
    assert open_replay["rollback"] is not None
    assert open_replay["rollback"]["reason"] == "open_frontier"

    forged = {**sealed, "receipt_id": "sha256:" + ("0" * 64)}
    bad = rollout.replay_decision_receipt(forged)
    assert bad["valid"] is False


def test_guide_documents_trust_safety_memory_transaction_recovery() -> None:
    text = _GUIDE_PATH.read_text(encoding="utf-8")
    lower = text.casefold()
    assert "shadow" in lower
    assert "assist" in lower
    assert "narrow-auto" in lower or "narrow_auto" in lower
    assert "rollback" in lower
    assert "memory safety" in lower
    assert "transaction" in lower
    assert "recovery" in lower
    assert "trust" in lower
    assert "fixed-point" in lower or "fixed point" in lower
    assert "doctor" in lower
    assert "replay" in lower
    assert any(p in lower for p in (
        "does not prove memory safety", "do not prove memory safety",
        "not memory-safety evidence", "not memory safety evidence",
    ))
    for topic in ("model-authored", "stateful", "cross-root", "generated", "dynamic", "native"):
        assert topic in lower or topic.replace("-", " ") in lower or topic.replace("-", "_") in lower
    result = rollout.check_guide_boundaries(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    for kind in ("vector", "test", "type", "resource", "max_memory_bytes", "llm"):
        assert rollout.evidence_proves_memory_safety(kind) is False
    statement = rollout.model_boundary_statement()
    assert "does not prove memory safety" in statement.casefold()
    trust = rollout.trust_boundary_statement()
    assert "trust" in trust.casefold()
    assert "recovery" in trust.casefold()


def test_cli_policy_and_check_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["policy", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "shadow"
    assert payload["mutation_authorized"] is False
    assert cli.main(["check-flags", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-rollback", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-dag", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-bindings", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-lanes", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"
    assert cli.main(["check-launcher", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "pass"


def test_cli_check_all_with_skips(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(["--check-all", "--skip-benchmark", "--json"])
    report = json.loads(capsys.readouterr().out)
    assert report["command"] == "check-all"
    assert report["default_mode"] == "shadow"
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["plan_objective_task_dag"] == "pass"
    assert statuses["exact_source_bindings"] == "pass"
    assert statuses["feature_flags"] == "pass"
    assert statuses["rollback_gates"] == "pass"
    assert statuses["guide_boundaries"] == "pass"
    assert statuses["benchmark_floors"] == "skip"
    assert code == 0
    assert report["valid"] is True


def test_cli_status(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv("LPR_STATE_ROOT", str(tmp_path / "stopped-program"))
    assert cli.main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "shadow"
    assert payload["valid"] is True


def test_cli_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    body = {
        "schema": "test/logic-repair-receipt@1",
        "transform": "add_argument", "unique_target": True, "reconstructed": True,
        "supported_python": True, "complete_frontier": True, "analytical_path": True,
        "fixed_point_ready": True,
        "roots": {"code_root": "sha256:abc"},
    }
    sealed = {**body, "receipt_id": rollout.content_identity(body)}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(sealed), encoding="utf-8")
    assert cli.main(["replay", "--receipt", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["automated_mutation_authorized"] is False


def test_operations_validator_facade(benchmark_report: dict) -> None:
    ops = rollout.LogicRepairOperationsValidator(_REPO_ROOT)
    report = ops.run_all(run_benchmark=False, probe_capabilities=True, benchmark_report=benchmark_report)
    assert report["valid"] is True
    assert ops.status()["mode"] == "shadow"
