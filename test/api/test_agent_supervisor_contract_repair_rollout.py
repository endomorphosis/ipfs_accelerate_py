"""RPR-020 / RPR-047: operations, metrics, feature flags, and rollback.

Acceptance (RPR-020 legacy preserved):

* Validation CLI checks plan/objective/task DAG, exact source bindings,
  capability health, supervisor/process state, and benchmark floors;
* shadow is default;
* assist and narrow-auto require explicit scoped policy;
* auto is initially limited to unique reconstructed supported
  substitutions/renames;
* capability regression, stale root, reconstruction failure, or metric breach
  rolls back;
* guide states model boundaries and that vector/test/type/resource evidence
  does not prove memory safety.

RPR-047 extension (without removing legacy checks):

* require RPR-G110/RPR-G220 and terminal RPR-047 with correct dependency chain;
* change_propagation_policy gates and six new zero safety floors;
* protected paths / refill isolation and exact source bindings;
* four-shard board drain readiness.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    RepairStrategy,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "validate_proof_gated_contract_repair.py"
_GUIDE_PATH = _REPO_ROOT / "docs" / "guides" / "PROOF_GATED_CONTRACT_REPAIR_GUIDE.md"
_BENCHMARK_PATH = _REPO_ROOT / "scripts" / "benchmark_contract_repair.py"


def _load_module():
    name = "validate_proof_gated_contract_repair_rpr020"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ops = _load_module()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_report() -> dict:
    """Reuse the hermetic RPR-019 benchmark once for floor checks."""

    if not _BENCHMARK_PATH.is_file():
        pytest.skip("benchmark script missing")
    name = "benchmark_contract_repair_rpr019_for_rpr020"
    if name in sys.modules:
        bench = sys.modules[name]
    else:
        spec = importlib.util.spec_from_file_location(name, _BENCHMARK_PATH)
        assert spec is not None and spec.loader is not None
        bench = importlib.util.module_from_spec(spec)
        sys.modules[name] = bench
        spec.loader.exec_module(bench)
    return bench.run_benchmark()


# ---------------------------------------------------------------------------
# Deliverables exist
# ---------------------------------------------------------------------------


def test_declared_outputs_exist() -> None:
    assert _SCRIPT_PATH.is_file()
    assert _GUIDE_PATH.is_file()
    assert Path(__file__).is_file()


def test_interfaces_and_schemas_are_stable() -> None:
    assert ops.ROLLOUT_POLICY_INTERFACE == "ContractRepairRolloutPolicy@1"
    assert ops.METRICS_INTERFACE == "ContractRepairMetrics@1"
    assert ops.VALIDATOR_INTERFACE == "ContractRepairValidatorOps@1"
    assert ops.TASK_ID == "RPR-020"
    assert ops.GOAL_ID == "RPR-G100"
    assert hasattr(ops, "ContractRepairRolloutPolicy")
    assert hasattr(ops, "ContractRepairMetrics")


# ---------------------------------------------------------------------------
# Shadow default / feature flags
# ---------------------------------------------------------------------------


def test_shadow_is_default_mode() -> None:
    policy = ops.default_rollout_policy()
    assert policy.mode_value == "shadow"
    assert policy.mode is ops.RolloutMode.SHADOW
    assert policy.mutation_authorized is False
    assert policy.completion_authoritative is False
    assert policy.allow_assist is False
    assert policy.allow_narrow_auto is False
    assert policy.allow_expanded_auto is False
    assert policy.auto_requires_unique_target is True
    assert policy.auto_requires_reconstruction is True
    assert policy.auto_requires_supported_shape is True


def test_assist_and_narrow_auto_require_explicit_scoped_policy() -> None:
    with pytest.raises(ops.ContractRepairValidationError, match="explicit scoped"):
        ops.ContractRepairRolloutPolicy(mode=ops.RolloutMode.ASSIST)
    with pytest.raises(ops.ContractRepairValidationError, match="explicit scoped"):
        ops.ContractRepairRolloutPolicy(mode=ops.RolloutMode.NARROW_AUTO)
    with pytest.raises(ops.ContractRepairValidationError, match="explicit scoped"):
        ops.ContractRepairRolloutPolicy(
            mode=ops.RolloutMode.ASSIST,
            explicit_policy_document="",  # empty document
            repository_id="repository:x",
        )

    assist = ops.elevate_rollout_policy(
        mode=ops.RolloutMode.ASSIST,
        explicit_policy_document="policy://reviewed/assist",
        repository_id="repository:demo",
    )
    assert assist.mode_value == "assist"
    assert assist.has_explicit_scoped_policy() is True
    assert assist.mutation_authorized is False

    narrow = ops.elevate_rollout_policy(
        mode=ops.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    assert narrow.mode_value == "narrow_auto"
    assert narrow.mutation_authorized is True


def test_auto_limited_to_unique_reconstructed_supported_renames() -> None:
    narrow = ops.elevate_rollout_policy(
        mode=ops.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    # Happy path: unique reconstructed supported rename/substitution.
    assert narrow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=True,
        reconstructed=True,
        supported_shape=True,
    )
    # Non-unique
    assert not narrow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=False,
        reconstructed=True,
        supported_shape=True,
    )
    # Unreconstructed
    assert not narrow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=True,
        reconstructed=False,
        supported_shape=True,
    )
    # Unsupported shape
    assert not narrow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=True,
        reconstructed=True,
        supported_shape=False,
    )
    # Adapter / new implementation not in initial auto set
    assert not narrow.allows_automated_mutation(
        strategy=RepairStrategy.ADAPTER.value,
        unique_target=True,
        reconstructed=True,
        supported_shape=True,
    )
    assert not narrow.allows_automated_mutation(
        strategy=RepairStrategy.NEW_IMPLEMENTATION.value,
        unique_target=True,
        reconstructed=True,
        supported_shape=True,
    )
    # Shadow never mutates
    shadow = ops.default_rollout_policy()
    assert not shadow.allows_automated_mutation(
        strategy=RepairStrategy.RENAME_SUBSTITUTION.value,
        unique_target=True,
        reconstructed=True,
        supported_shape=True,
    )


def test_feature_flag_check_passes() -> None:
    result = ops.check_feature_flags()
    assert result.status is ops.CheckStatus.PASS
    assert result.ok


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_capability_regression_stale_root_reconstruction_metric_roll_back() -> None:
    policy = ops.elevate_rollout_policy(
        mode=ops.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )

    cap = ops.evaluate_rollback(
        policy, capability_regression=("logic_smt", "cvc5")
    )
    assert cap is not None
    assert cap.reason is ops.RollbackReason.CAPABILITY_REGRESSION
    demoted = ops.apply_rollback(policy, cap)
    assert demoted.mutation_authorized is False
    assert demoted.mode_value in {"assist", "shadow"}

    stale = ops.evaluate_rollback(policy, stale_roots=("index_root", "tree_id"))
    assert stale is not None
    assert stale.reason is ops.RollbackReason.STALE_ROOT

    recon = ops.evaluate_rollback(policy, reconstruction_failed=True)
    assert recon is not None
    assert recon.reason is ops.RollbackReason.RECONSTRUCTION_FAILURE

    breached = ops.ContractRepairMetrics(
        wrong_path_rate=1,
        safety_floors={
            "wrong_path_automated_mutation_rate": 1,
            "failed_obligation_override_rate": 0,
            "stale_forged_or_poisoned_authoritative_admission_rate": 0,
            "unsupported_memory_safety_promotion_rate": 0,
        },
        safety_absolute={"wrong_path_automated_mutation": 1},
    )
    metric = ops.evaluate_rollback(policy, metrics=breached)
    assert metric is not None
    assert metric.reason is ops.RollbackReason.METRIC_BREACH
    assert "wrong_path" in " ".join(metric.metric_breaches)

    healthy = ops.evaluate_rollback(
        policy, metrics=ops.ContractRepairMetrics.empty()
    )
    assert healthy is None


def test_rollback_gate_check_passes() -> None:
    result = ops.check_rollback_gates()
    assert result.status is ops.CheckStatus.PASS


# ---------------------------------------------------------------------------
# Validation CLI checks
# ---------------------------------------------------------------------------


def test_plan_objective_task_dag_check() -> None:
    result = ops.check_plan_objective_task_dag(_REPO_ROOT)
    assert result.status is ops.CheckStatus.PASS, result.detail
    assert result.evidence["task_count"] >= 20
    assert "RPR-G100" in result.evidence["goal_ids"]
    assert "RPR-020" in result.evidence["task_ids"]
    # RPR-047 extension IDs remain required alongside legacy checks.
    assert "RPR-G110" in result.evidence["goal_ids"]
    assert "RPR-G220" in result.evidence["goal_ids"]
    assert "RPR-047" in result.evidence["task_ids"]
    assert result.evidence["terminal_task_id"] == "RPR-047"
    assert result.evidence["extension_goal_ids"] == ["RPR-G110", "RPR-G220"]
    floors = result.evidence["propagation_safety_floors"]
    for key in ops.PROPAGATION_SAFETY_FLOOR_KEYS:
        assert floors[key] == 0
    policy = result.evidence["change_propagation_policy"]
    for key, expected in ops.REQUIRED_CHANGE_PROPAGATION_POLICY.items():
        assert policy[key] == expected


def test_exact_source_bindings_check() -> None:
    result = ops.check_exact_source_bindings(_REPO_ROOT)
    assert result.status is ops.CheckStatus.PASS, result.detail
    binding = ops.bind_exact_sources(_REPO_ROOT)
    assert binding.board_namespace == ops.BOARD_NAMESPACE
    assert binding.task_prefix == ops.TASK_PREFIX
    assert binding.plan_identity.startswith("sha256:")
    assert binding.binding_id.startswith("sha256:")
    # Re-bind is stable for unchanged files.
    again = ops.bind_exact_sources(_REPO_ROOT)
    assert again.binding_id == binding.binding_id


def test_capability_health_check() -> None:
    result = ops.check_capability_health(_REPO_ROOT, probe=True)
    assert result.status is ops.CheckStatus.PASS, result.detail
    assert result.evidence.get("authoritative") is False
    assert result.evidence.get("candidate_authoritative") is False


def test_supervisor_process_state_check(tmp_path: Path) -> None:
    # Empty state root: stopped master is healthy.
    result = ops.check_supervisor_process_state(
        _REPO_ROOT, state_root=tmp_path / "missing-program"
    )
    assert result.status is ops.CheckStatus.PASS, result.detail
    assert result.evidence["master_status"] == "stopped"

    # Contradictory running + dead pid fails closed.
    program = tmp_path / "program"
    lane0 = program / "state" / "lane-0"
    lane0.mkdir(parents=True)
    (lane0 / "rpr_lane_0_supervisor_status.json").write_text(
        json.dumps({"status": "running", "pid": 2**30, "supervisor_pid": 2**30}),
        encoding="utf-8",
    )
    (lane0 / "rpr_lane_0_task_state.json").write_text(
        json.dumps(
            {
                "status": "available",
                "active_task_id": "",
                "eligible_ready_count": 0,
                "blocked_count": 0,
            }
        ),
        encoding="utf-8",
    )
    bad = ops.check_supervisor_process_state(
        _REPO_ROOT, state_root=program, lane_count=1
    )
    assert bad.status is ops.CheckStatus.FAIL
    assert "dead" in bad.detail.casefold() or "running" in bad.detail.casefold()


def test_benchmark_floors_check(benchmark_report: dict) -> None:
    result = ops.check_benchmark_floors(
        _REPO_ROOT, run=False, report=benchmark_report
    )
    assert result.status is ops.CheckStatus.PASS, result.detail
    floors = result.evidence["safety_floors"]
    for key in ops.SAFETY_FLOOR_KEYS:
        assert floors[key] == 0


def test_run_all_checks_with_cached_benchmark(benchmark_report: dict) -> None:
    report = ops.run_all_checks(
        _REPO_ROOT,
        run_benchmark=False,
        probe_capabilities=True,
        benchmark_report=benchmark_report,
    )
    assert report["schema"] == ops.VALIDATOR_SCHEMA
    assert report["default_mode"] == "shadow"
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False
    assert report["valid"] is True, report.get("failed")
    assert report["terminal_task_id"] == "RPR-047"
    assert report["extension_goal_ids"] == ["RPR-G110", "RPR-G220"]
    names = {item["name"] for item in report["checks"]}
    assert {
        "plan_objective_task_dag",
        "exact_source_bindings",
        "capability_health",
        "supervisor_process_state",
        "benchmark_floors",
        "feature_flags",
        "rollback_gates",
        "guide_boundaries",
        "change_propagation_policy",
        "protected_paths_refill_isolation",
        "four_shard_board_drain",
    } <= names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_contract_repair_metrics_from_benchmark(benchmark_report: dict) -> None:
    metrics = ops.ContractRepairMetrics.from_benchmark_metrics(
        benchmark_report["metrics"]
    )
    assert metrics.INTERFACE == "ContractRepairMetrics@1"
    assert metrics.floors_hold()
    assert metrics.wrong_path_rate == 0
    assert metrics.decision_count == benchmark_report["metrics"]["case_count"]
    assert metrics.recall_at_k == benchmark_report["metrics"]["recall_at_k"]
    assert (
        metrics.proof_eligible_recall_at_k
        == benchmark_report["metrics"]["proof_eligible_recall_at_k"]
    )
    assert (
        metrics.admitted_precision
        == benchmark_report["metrics"]["admitted_target_precision"]
    )
    assert metrics.abstention_count == benchmark_report["metrics"]["abstention_count"]
    assert metrics.tokens == benchmark_report["metrics"]["total_token_units"]
    assert metrics.context_bytes == benchmark_report["metrics"]["total_context_bytes"]
    assert metrics.cache_hit_rate == benchmark_report["metrics"]["cache_hit_rate"]
    assert metrics.metrics_id.startswith("sha256:")
    # Deterministic seal
    again = ops.ContractRepairMetrics.from_benchmark_metrics(
        benchmark_report["metrics"]
    )
    assert again.metrics_id == metrics.metrics_id
    payload = metrics.to_dict()
    for key in (
        "recall_at_k",
        "proof_eligible_recall_at_k",
        "admitted_precision",
        "wrong_path_rate",
        "abstention_count",
        "proof_latency_ms",
        "cache_latency_ms",
        "tokens",
        "context_bytes",
    ):
        assert key in payload


def test_collect_metrics_empty_without_run() -> None:
    metrics = ops.collect_metrics(run_benchmark=False)
    assert metrics.floors_hold()
    assert metrics.decision_count == 0


# ---------------------------------------------------------------------------
# Doctor / status / replay
# ---------------------------------------------------------------------------


def test_doctor_and_status_commands(benchmark_report: dict) -> None:
    doctor = ops.doctor(
        _REPO_ROOT, run_benchmark=False, probe_capabilities=True
    )
    # doctor uses run_all_checks with run_benchmark=False, so floors may skip;
    # force floors via full check with cached report for completeness.
    full = ops.run_all_checks(
        _REPO_ROOT,
        run_benchmark=False,
        probe_capabilities=True,
        benchmark_report=benchmark_report,
    )
    assert full["valid"] is True
    assert doctor["command"] == "doctor"
    assert doctor["default_mode"] == "shadow"

    st = ops.status(_REPO_ROOT)
    assert st["command"] == "status"
    assert st["mode"] == "shadow"
    assert st["valid"] is True
    assert st["supervisor"]["evidence"]["master_status"] in {"running", "stopped"}


def test_replay_receipt_identity_and_auto_gate() -> None:
    body = {
        "schema": "test/receipt@1",
        "strategy": RepairStrategy.RENAME_SUBSTITUTION.value,
        "unique_target": True,
        "reconstructed": True,
        "supported_shape": True,
        "language": "python",
        "roots": {"code_root": "sha256:abc", "index_root": "sha256:def"},
    }
    sealed = {**body, "receipt_id": ops.content_identity(body)}

    # Shadow: valid identity, no auto mutation.
    shadow_replay = ops.replay_decision_receipt(sealed)
    assert shadow_replay["valid"] is True
    assert shadow_replay["automated_mutation_authorized"] is False

    narrow = ops.elevate_rollout_policy(
        mode=ops.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    auto_replay = ops.replay_decision_receipt(sealed, policy=narrow)
    assert auto_replay["valid"] is True
    assert auto_replay["automated_mutation_authorized"] is True

    # Stale root forces rollback evaluation under expected roots.
    stale_replay = ops.replay_decision_receipt(
        sealed,
        policy=narrow,
        expected_roots={"code_root": "sha256:other"},
    )
    assert stale_replay["valid"] is False
    assert stale_replay["rollback"] is not None
    assert stale_replay["rollback"]["reason"] == "stale_root"

    # Forged identity fails.
    forged = {**sealed, "receipt_id": "sha256:" + ("0" * 64)}
    bad = ops.replay_decision_receipt(forged)
    assert bad["valid"] is False


# ---------------------------------------------------------------------------
# Guide boundaries
# ---------------------------------------------------------------------------


def test_guide_states_model_boundaries_and_non_memory_safety_evidence() -> None:
    text = _GUIDE_PATH.read_text(encoding="utf-8")
    lower = text.casefold()
    assert "shadow" in lower
    assert "assist" in lower
    assert "narrow-auto" in lower or "narrow_auto" in lower
    assert "rollback" in lower
    assert "memory safety" in lower
    assert "vector" in lower
    assert "test" in lower
    assert "type" in lower
    assert "resource" in lower
    assert any(
        phrase in lower
        for phrase in (
            "does not prove memory safety",
            "do not prove memory safety",
            "not memory-safety evidence",
            "not memory safety evidence",
        )
    )
    assert "models propose" in lower or "model" in lower

    result = ops.check_guide_boundaries(_REPO_ROOT)
    assert result.status is ops.CheckStatus.PASS, result.detail

    # Helper encodes the negative rule for non-proof evidence kinds.
    for kind in ("vector", "test", "type", "resource", "max_memory_bytes", "llm"):
        assert ops.evidence_proves_memory_safety(kind) is False
    statement = ops.model_boundary_statement()
    assert "does not prove memory safety" in statement.casefold()
    assert "vector" in statement.casefold()


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def test_cli_policy_and_check_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert ops.main(["policy", "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "shadow"
    assert payload["mutation_authorized"] is False

    assert ops.main(["check-flags", "--json"]) == 0
    flags = json.loads(capsys.readouterr().out)
    assert flags["status"] == "pass"

    assert ops.main(["check-rollback", "--json"]) == 0
    rb = json.loads(capsys.readouterr().out)
    assert rb["status"] == "pass"

    assert ops.main(["check-dag", "--json"]) == 0
    dag = json.loads(capsys.readouterr().out)
    assert dag["status"] == "pass"

    assert ops.main(["check-bindings", "--json"]) == 0
    bindings = json.loads(capsys.readouterr().out)
    assert bindings["status"] == "pass"


def test_cli_check_all_with_skips(capsys: pytest.CaptureFixture[str]) -> None:
    # Skip heavy benchmark for speed; floors still covered via unit tests above.
    code = ops.main(
        [
            "--check-all",
            "--skip-benchmark",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    report = json.loads(out)
    # guide + flags + dag + bindings + supervisor should pass; benchmark skipped.
    assert report["command"] == "check-all"
    assert report["default_mode"] == "shadow"
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["plan_objective_task_dag"] == "pass"
    assert statuses["exact_source_bindings"] == "pass"
    assert statuses["feature_flags"] == "pass"
    assert statuses["rollback_gates"] == "pass"
    assert statuses["guide_boundaries"] == "pass"
    assert statuses["benchmark_floors"] == "skip"
    assert statuses["change_propagation_policy"] == "pass"
    assert statuses["protected_paths_refill_isolation"] == "pass"
    assert statuses["four_shard_board_drain"] == "pass"
    # valid is true when only skips remain among non-fail checks
    assert code == 0
    assert report["valid"] is True


# ---------------------------------------------------------------------------
# RPR-047 extension gates
# ---------------------------------------------------------------------------


def test_change_propagation_policy_and_six_new_floors() -> None:
    result = ops.check_change_propagation_policy(_REPO_ROOT)
    assert result.status is ops.CheckStatus.PASS, result.detail
    floors = result.evidence["propagation_safety_floors"]
    assert len(ops.PROPAGATION_SAFETY_FLOOR_KEYS) == 6
    for key in ops.PROPAGATION_SAFETY_FLOOR_KEYS:
        assert floors[key] == 0
    for key, expected in ops.REQUIRED_CHANGE_PROPAGATION_POLICY.items():
        assert result.evidence["change_propagation_policy"][key] == expected


def test_protected_paths_and_refill_isolation() -> None:
    result = ops.check_protected_paths_and_refill_isolation(_REPO_ROOT)
    assert result.status is ops.CheckStatus.PASS, result.detail
    protected = set(result.evidence["protected_paths"])
    for path in ops.REQUIRED_PROTECTED_PATHS:
        assert path in protected
        assert (_REPO_ROOT / path).is_file()
    assert result.evidence["objective_refill_enabled"] is False
    assert result.evidence["codebase_refill_enabled"] is False


def test_four_shard_board_can_drain() -> None:
    result = ops.check_four_shard_board_drain(_REPO_ROOT)
    assert result.status is ops.CheckStatus.PASS, result.detail
    assert result.evidence["max_lanes"] == 4
    assert result.evidence["strict_task_sharding"] is True
    assert result.evidence["terminal_task_id"] == "RPR-047"
    assert result.evidence["ready_after_non_terminal_complete"] == ["RPR-047"]
    assert result.evidence["ready_after_full_complete"] == []
    lanes = result.evidence["entry_lanes"]
    assert set(lanes) == {"RPR-022", "RPR-023", "RPR-024", "RPR-025"}
    assert len(set(lanes.values())) == 4


def test_proof_gated_operations_surface(benchmark_report: dict) -> None:
    assert ops.ProofGatedContractRepairOperations.INTERFACE == (
        "ProofGatedContractRepairOperations@1"
    )
    ids = ops.ProofGatedContractRepairOperations.required_extension_ids()
    assert ids["terminal_task_id"] == "RPR-047"
    assert ids["control_goal_id"] == "RPR-G110"
    assert ids["rollout_goal_id"] == "RPR-G220"
    assert ids["legacy_task_id"] == "RPR-020"
    assert ids["legacy_goal_id"] == "RPR-G100"
    report = ops.ProofGatedContractRepairOperations.run(
        _REPO_ROOT,
        run_benchmark=False,
        probe_capabilities=True,
        benchmark_report=benchmark_report,
    )
    assert report["valid"] is True, report.get("failed")
    assert report["terminal_task_id"] == "RPR-047"
    assert report["legacy_task_id"] == "RPR-020"
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False


def test_cli_propagation_extension_checks(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert ops.main(["check-propagation-policy", "--json"]) == 0
    policy = json.loads(capsys.readouterr().out)
    assert policy["status"] == "pass"

    assert ops.main(["check-protected-paths", "--json"]) == 0
    protected = json.loads(capsys.readouterr().out)
    assert protected["status"] == "pass"

    assert ops.main(["check-four-shard", "--json"]) == 0
    shards = json.loads(capsys.readouterr().out)
    assert shards["status"] == "pass"


def test_cli_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert ops.main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "shadow"
    assert payload["valid"] is True


def test_cli_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    body = {
        "schema": "test/receipt@1",
        "strategy": "rename_substitution",
        "unique_target": True,
        "reconstructed": True,
        "supported_shape": True,
    }
    receipt = {**body, "receipt_id": ops.content_identity(body)}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    assert ops.main(["replay", "--receipt", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["automated_mutation_authorized"] is False


def test_policy_round_trip_dict() -> None:
    original = ops.elevate_rollout_policy(
        mode=ops.RolloutMode.ASSIST,
        explicit_policy_document="policy://reviewed/assist",
        repository_id="repository:roundtrip",
        scoped_path_globs=("src/**",),
    )
    restored = ops.ContractRepairRolloutPolicy.from_dict(original.to_dict())
    assert restored.mode_value == original.mode_value
    assert restored.policy_binding_id == original.policy_binding_id
    assert restored.scoped_path_globs == original.scoped_path_globs


def test_shadow_cannot_authorize_mutation() -> None:
    with pytest.raises(ops.ContractRepairValidationError, match="shadow"):
        ops.ContractRepairRolloutPolicy(mutation_authorized=True)


def test_checkpoint_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_TASK_CHECKPOINT_DIR", str(tmp_path))
    path = ops.write_checkpoint("rpr-020-test", {"ok": True, "n": 1})
    assert path is not None
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["ok"] is True
