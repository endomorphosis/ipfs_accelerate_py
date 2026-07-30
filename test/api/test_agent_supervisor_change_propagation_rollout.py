"""RPR-046: propagation metrics, rollout flags, CLI, guide, and rollback.

Acceptance:

* CLI validates exact sources, capabilities, graph/index coverage, proof
  reconstruction, transaction health and benchmark floors;
* metrics expose every benchmark stage plus analytical/model split, tokens/
  context and fixed-point iterations;
* shadow is default;
* assist requires explicit policy;
* narrow-auto is limited to complete-frontier unique reconstructed analytical
  supported-Python transforms;
* model-authored, stateful behavior, public schema/API, dynamic/generated/
  native and cross-root changes remain approval-gated;
* stale roots, open frontier, capability regression, proof loss, wrong-value/
  missed-consumer/partial-plan/false-completion or any floor breach rolls back;
* guide documents trust, safety, memory, transaction and recovery boundaries.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.change_propagation_contracts import (
    TransformKind,
)
from ipfs_accelerate_py.agent_supervisor.validation import change_propagation_rollout as rollout

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "validate_change_propagation.py"
_GUIDE_PATH = _REPO_ROOT / "docs" / "guides" / "PROOF_GATED_CHANGE_PROPAGATION_GUIDE.md"
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "change_propagation_rollout.py"
)
_BENCHMARK_PATH = _REPO_ROOT / "scripts" / "benchmark_change_propagation.py"


def _load_cli():
    name = "validate_change_propagation_rpr046"
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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def benchmark_report() -> dict:
    """Reuse the hermetic RPR-045 benchmark once for floor checks."""

    if not _BENCHMARK_PATH.is_file():
        pytest.skip("benchmark script missing")
    name = "benchmark_change_propagation_rpr045_for_rpr046"
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
    assert _MODULE_PATH.is_file()
    assert _SCRIPT_PATH.is_file()
    assert _GUIDE_PATH.is_file()
    assert Path(__file__).is_file()


def test_interfaces_and_schemas_are_stable() -> None:
    assert rollout.ROLLOUT_POLICY_INTERFACE == "ChangePropagationRolloutPolicy@1"
    assert rollout.METRICS_INTERFACE == "ChangePropagationMetrics@1"
    assert rollout.ROLLBACK_GATE_INTERFACE == "ChangePropagationRollbackGate@1"
    assert rollout.VALIDATOR_INTERFACE == "ChangePropagationValidatorOps@1"
    assert rollout.BENCHMARK_METRICS_INTERFACE == "ChangePropagationBenchmarkMetrics@1"
    assert rollout.ATOMIC_PROPAGATION_PLAN_INTERFACE == "AtomicPropagationPlan@1"
    assert (
        rollout.PROPAGATION_COMPLETION_RECEIPT_INTERFACE
        == "PropagationCompletionReceipt@1"
    )
    assert rollout.TASK_ID == "RPR-046"
    assert rollout.GOAL_ID == "RPR-G220"
    assert hasattr(rollout, "ChangePropagationRolloutPolicy")
    assert hasattr(rollout, "ChangePropagationMetrics")
    assert hasattr(rollout, "ChangePropagationRollbackGate")


# ---------------------------------------------------------------------------
# Shadow default / feature flags
# ---------------------------------------------------------------------------


def test_shadow_is_default_mode() -> None:
    policy = rollout.default_rollout_policy()
    assert policy.mode_value == "shadow"
    assert policy.mode is rollout.RolloutMode.SHADOW
    assert policy.mutation_authorized is False
    assert policy.completion_authoritative is False
    assert policy.allow_assist is False
    assert policy.allow_narrow_auto is False
    assert policy.allow_expanded_auto is False
    assert policy.auto_requires_unique_target is True
    assert policy.auto_requires_reconstruction is True
    assert policy.auto_requires_supported_python is True
    assert policy.auto_requires_complete_frontier is True
    assert policy.auto_requires_analytical_path is True


def test_assist_requires_explicit_policy() -> None:
    with pytest.raises(rollout.ChangePropagationRolloutError, match="explicit scoped"):
        rollout.ChangePropagationRolloutPolicy(mode=rollout.RolloutMode.ASSIST)
    with pytest.raises(rollout.ChangePropagationRolloutError, match="explicit scoped"):
        rollout.ChangePropagationRolloutPolicy(mode=rollout.RolloutMode.NARROW_AUTO)
    with pytest.raises(rollout.ChangePropagationRolloutError, match="explicit scoped"):
        rollout.ChangePropagationRolloutPolicy(
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


def test_narrow_auto_limited_to_complete_frontier_analytical_python() -> None:
    narrow = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    # Happy path.
    assert narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )
    assert narrow.allows_automated_mutation(
        transform=TransformKind.THREAD_PARAMETER.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )
    # Incomplete frontier
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=False,
        analytical_path=True,
    )
    # Non-unique
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.RENAME_ARGUMENT.value,
        unique_target=False,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )
    # Unreconstructed
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.REORDER_ARGUMENT.value,
        unique_target=True,
        reconstructed=False,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )
    # Non-Python shape
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=False,
        complete_frontier=True,
        analytical_path=True,
    )
    # Non-analytical path
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=False,
    )
    # Shadow never mutates
    shadow = rollout.default_rollout_policy()
    assert not shadow.allows_automated_mutation(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )


def test_approval_gated_families_remain_blocked() -> None:
    narrow = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )
    base = dict(
        transform=TransformKind.ADD_ARGUMENT.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )
    assert not narrow.allows_automated_mutation(**base, model_authored=True)
    assert not narrow.allows_automated_mutation(**base, stateful=True)
    assert not narrow.allows_automated_mutation(**base, public_schema_api=True)
    assert not narrow.allows_automated_mutation(**base, dynamic=True)
    assert not narrow.allows_automated_mutation(**base, generated=True)
    assert not narrow.allows_automated_mutation(**base, native=True)
    assert not narrow.allows_automated_mutation(**base, cross_root=True)
    # Schema / generated transform kinds are gated by family membership.
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.UPDATE_SCHEMA_FIELD.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.UPDATE_GENERATED_MANIFEST.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )
    assert not narrow.allows_automated_mutation(
        transform=TransformKind.ADD_ADAPTER.value,
        unique_target=True,
        reconstructed=True,
        supported_python=True,
        complete_frontier=True,
        analytical_path=True,
    )


def test_feature_flag_check_passes() -> None:
    result = rollout.check_feature_flags()
    assert result.status is rollout.CheckStatus.PASS
    assert result.ok


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


def test_rollback_triggers_demote_and_revoke_mutation() -> None:
    policy = rollout.elevate_rollout_policy(
        mode=rollout.RolloutMode.NARROW_AUTO,
        explicit_policy_document="policy://reviewed/narrow-auto",
        repository_id="repository:demo",
        mutation_authorized=True,
    )

    cap = rollout.evaluate_rollback(
        policy, capability_regression=("graph_index", "logic_smt")
    )
    assert cap is not None
    assert cap.reason is rollout.RollbackReason.CAPABILITY_REGRESSION
    demoted = rollout.apply_rollback(policy, cap)
    assert demoted.mutation_authorized is False
    assert demoted.mode_value in {"assist", "shadow"}

    stale = rollout.evaluate_rollback(policy, stale_roots=("index_root", "tree_id"))
    assert stale is not None
    assert stale.reason is rollout.RollbackReason.STALE_ROOT

    open_f = rollout.evaluate_rollback(policy, open_frontier=True)
    assert open_f is not None
    assert open_f.reason is rollout.RollbackReason.OPEN_FRONTIER

    recon = rollout.evaluate_rollback(policy, reconstruction_failed=True)
    assert recon is not None
    assert recon.reason is rollout.RollbackReason.RECONSTRUCTION_FAILURE

    proof = rollout.evaluate_rollback(policy, proof_loss=True)
    assert proof is not None
    assert proof.reason is rollout.RollbackReason.PROOF_LOSS

    wrong = rollout.evaluate_rollback(policy, wrong_value=True)
    assert wrong is not None
    assert wrong.reason is rollout.RollbackReason.WRONG_VALUE

    missed = rollout.evaluate_rollback(policy, missed_consumer=True)
    assert missed is not None
    assert missed.reason is rollout.RollbackReason.MISSED_CONSUMER

    partial = rollout.evaluate_rollback(policy, partial_plan=True)
    assert partial is not None
    assert partial.reason is rollout.RollbackReason.PARTIAL_PLAN

    false_c = rollout.evaluate_rollback(policy, false_completion=True)
    assert false_c is not None
    assert false_c.reason is rollout.RollbackReason.FALSE_COMPLETION

    breached = rollout.ChangePropagationMetrics(
        wrong_path_rate=1,
        safety_floors={
            **{key: 0 for key in rollout.SAFETY_FLOOR_KEYS},
            "wrong_path_automated_mutation_rate": 1,
        },
        safety_absolute={
            **{key: 0 for key in rollout.SAFETY_ABSOLUTE_KEYS},
            "wrong_path_automated_mutation": 1,
        },
    )
    metric = rollout.evaluate_rollback(policy, metrics=breached)
    assert metric is not None
    assert metric.reason is rollout.RollbackReason.METRIC_BREACH
    assert "wrong_path" in " ".join(metric.metric_breaches)

    healthy = rollout.evaluate_rollback(
        policy, metrics=rollout.ChangePropagationMetrics.empty()
    )
    assert healthy is None


def test_rollback_gate_class_and_check_pass() -> None:
    gate = rollout.ChangePropagationRollbackGate()
    assert gate.INTERFACE == "ChangePropagationRollbackGate@1"
    assert gate.evaluate(stale_roots=()) is None
    result = rollout.check_rollback_gates()
    assert result.status is rollout.CheckStatus.PASS


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------


def test_plan_objective_task_dag_check() -> None:
    result = rollout.check_plan_objective_task_dag(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["task_count"] >= 20
    assert "RPR-G220" in result.evidence["goal_ids"]
    assert "RPR-046" in result.evidence["task_ids"]


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
    again = rollout.bind_exact_sources(_REPO_ROOT)
    assert again.binding_id == binding.binding_id


def test_capability_health_check() -> None:
    result = rollout.check_capability_health(_REPO_ROOT, probe=True)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence.get("authoritative") is False
    assert result.evidence.get("candidate_authoritative") is False


def test_graph_index_coverage_check() -> None:
    result = rollout.check_graph_index_coverage(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["case_count"] >= 1
    assert result.evidence["modules"]


def test_proof_reconstruction_check() -> None:
    result = rollout.check_proof_reconstruction(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["auto_requires_reconstruction"] is True


def test_transaction_health_check() -> None:
    result = rollout.check_transaction_health(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["partial_groups_cannot_merge"] is True


def test_supervisor_process_state_check(tmp_path: Path) -> None:
    result = rollout.check_supervisor_process_state(
        _REPO_ROOT, state_root=tmp_path / "missing-program"
    )
    assert result.status is rollout.CheckStatus.PASS, result.detail
    assert result.evidence["master_status"] == "stopped"

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
    bad = rollout.check_supervisor_process_state(
        _REPO_ROOT, state_root=program, lane_count=1
    )
    assert bad.status is rollout.CheckStatus.FAIL
    assert "dead" in bad.detail.casefold() or "running" in bad.detail.casefold()


def test_benchmark_floors_check(benchmark_report: dict) -> None:
    result = rollout.check_benchmark_floors(
        _REPO_ROOT, run=False, report=benchmark_report
    )
    assert result.status is rollout.CheckStatus.PASS, result.detail
    floors = result.evidence["safety_floors"]
    for key in rollout.SAFETY_FLOOR_KEYS:
        assert floors[key] == 0
    assert result.evidence["fixed_point_iterations_total"] >= 0
    assert set(result.evidence["benchmark_stages"]) == set(rollout.BENCHMARK_STAGES)


def test_run_all_checks_with_cached_benchmark(benchmark_report: dict) -> None:
    report = rollout.run_all_checks(
        _REPO_ROOT,
        run_benchmark=False,
        probe_capabilities=True,
        benchmark_report=benchmark_report,
    )
    assert report["schema"] == rollout.VALIDATOR_SCHEMA
    assert report["default_mode"] == "shadow"
    assert report["mutation_authorized"] is False
    assert report["completion_authoritative"] is False
    assert report["valid"] is True, report.get("failed")
    names = {item["name"] for item in report["checks"]}
    assert {
        "plan_objective_task_dag",
        "exact_source_bindings",
        "capability_health",
        "graph_index_coverage",
        "proof_reconstruction",
        "transaction_health",
        "supervisor_process_state",
        "benchmark_floors",
        "feature_flags",
        "rollback_gates",
        "guide_boundaries",
    } <= names


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_expose_stages_split_tokens_fixed_point(
    benchmark_report: dict,
) -> None:
    metrics = rollout.ChangePropagationMetrics.from_benchmark_metrics(
        benchmark_report["metrics"]
    )
    assert metrics.INTERFACE == "ChangePropagationMetrics@1"
    assert metrics.floors_hold()
    assert metrics.case_count == benchmark_report["metrics"]["case_count"]
    assert metrics.tokens == benchmark_report["metrics"]["total_token_units"]
    assert metrics.context_bytes == benchmark_report["metrics"]["total_context_bytes"]
    assert (
        metrics.fixed_point_iterations_total
        == benchmark_report["metrics"]["fixed_point_iterations_total"]
    )
    assert metrics.fixed_point_iterations == metrics.fixed_point_iterations_total
    assert metrics.analytical_coverage == benchmark_report["metrics"][
        "analytical_coverage"
    ]
    assert metrics.model_rate == benchmark_report["metrics"]["llm_rate"]
    assert metrics.llm_rate == benchmark_report["metrics"]["llm_rate"]
    split = metrics.analytical_model_split
    assert "analytical_coverage" in split
    assert "model_rate" in split or "llm_rate" in split
    for stage in rollout.BENCHMARK_STAGES:
        assert stage in metrics.stage_counts
        assert stage in metrics.stage_cost_units
    payload = metrics.to_dict()
    for key in (
        "benchmark_stages",
        "analytical_model_split",
        "tokens",
        "context_bytes",
        "fixed_point_iterations",
        "fixed_point_iterations_total",
        "stage_counts",
        "safety_floors",
    ):
        assert key in payload
    assert set(payload["benchmark_stages"]) == set(rollout.BENCHMARK_STAGES)
    again = rollout.ChangePropagationMetrics.from_benchmark_metrics(
        benchmark_report["metrics"]
    )
    assert again.metrics_id == metrics.metrics_id
    assert metrics.metrics_id.startswith("sha256:")


def test_collect_metrics_empty_without_run() -> None:
    metrics = rollout.collect_metrics(run_benchmark=False)
    assert metrics.floors_hold()
    assert metrics.case_count == 0
    for stage in rollout.BENCHMARK_STAGES:
        assert stage in metrics.stage_counts


# ---------------------------------------------------------------------------
# Doctor / status / replay
# ---------------------------------------------------------------------------


def test_doctor_and_status_commands(benchmark_report: dict) -> None:
    doctor = rollout.doctor(
        _REPO_ROOT, run_benchmark=False, probe_capabilities=True
    )
    full = rollout.run_all_checks(
        _REPO_ROOT,
        run_benchmark=False,
        probe_capabilities=True,
        benchmark_report=benchmark_report,
    )
    assert full["valid"] is True
    assert doctor["command"] == "doctor"
    assert doctor["default_mode"] == "shadow"

    st = rollout.status(_REPO_ROOT)
    assert st["command"] == "status"
    assert st["mode"] == "shadow"
    assert st["valid"] is True
    assert st["supervisor"]["evidence"]["master_status"] in {"running", "stopped"}
    assert st["graph_index_coverage"]["status"] == "pass"
    assert st["transaction_health"]["status"] == "pass"


def test_replay_receipt_identity_and_auto_gate() -> None:
    body = {
        "schema": "test/propagation-receipt@1",
        "transform": TransformKind.ADD_ARGUMENT.value,
        "unique_target": True,
        "reconstructed": True,
        "supported_python": True,
        "complete_frontier": True,
        "analytical_path": True,
        "language": "python",
        "roots": {"code_root": "sha256:abc", "index_root": "sha256:def"},
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
        sealed,
        policy=narrow,
        expected_roots={"code_root": "sha256:other"},
    )
    assert stale_replay["valid"] is False
    assert stale_replay["rollback"] is not None
    assert stale_replay["rollback"]["reason"] == "stale_root"

    open_body = {
        **body,
        "complete_frontier": False,
        "open_frontier": True,
    }
    open_sealed = {**open_body, "receipt_id": rollout.content_identity(open_body)}
    open_replay = rollout.replay_decision_receipt(open_sealed, policy=narrow)
    assert open_replay["automated_mutation_authorized"] is False
    assert open_replay["rollback"] is not None
    assert open_replay["rollback"]["reason"] == "open_frontier"

    forged = {**sealed, "receipt_id": "sha256:" + ("0" * 64)}
    bad = rollout.replay_decision_receipt(forged)
    assert bad["valid"] is False


# ---------------------------------------------------------------------------
# Guide boundaries
# ---------------------------------------------------------------------------


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
    assert "complete frontier" in lower
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
    for topic in (
        "model-authored",
        "stateful",
        "cross-root",
        "generated",
        "dynamic",
        "native",
    ):
        assert topic in lower or topic.replace("-", " ") in lower or topic.replace(
            "-", "_"
        ) in lower

    result = rollout.check_guide_boundaries(_REPO_ROOT)
    assert result.status is rollout.CheckStatus.PASS, result.detail

    for kind in ("vector", "test", "type", "resource", "max_memory_bytes", "llm"):
        assert rollout.evidence_proves_memory_safety(kind) is False
    statement = rollout.model_boundary_statement()
    assert "does not prove memory safety" in statement.casefold()
    assert "complete-frontier" in statement.casefold() or "complete frontier" in statement.casefold()
    trust = rollout.trust_boundary_statement()
    assert "trust" in trust.casefold()
    assert "recovery" in trust.casefold()


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------


def test_cli_policy_and_check_flags(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["policy", "--json"]) == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["mode"] == "shadow"
    assert payload["mutation_authorized"] is False

    assert cli.main(["check-flags", "--json"]) == 0
    flags = json.loads(capsys.readouterr().out)
    assert flags["status"] == "pass"

    assert cli.main(["check-rollback", "--json"]) == 0
    rb = json.loads(capsys.readouterr().out)
    assert rb["status"] == "pass"

    assert cli.main(["check-dag", "--json"]) == 0
    dag = json.loads(capsys.readouterr().out)
    assert dag["status"] == "pass"

    assert cli.main(["check-bindings", "--json"]) == 0
    bindings = json.loads(capsys.readouterr().out)
    assert bindings["status"] == "pass"

    assert cli.main(["check-graph-index", "--json"]) == 0
    graph = json.loads(capsys.readouterr().out)
    assert graph["status"] == "pass"

    assert cli.main(["check-proof-reconstruction", "--json"]) == 0
    proof = json.loads(capsys.readouterr().out)
    assert proof["status"] == "pass"

    assert cli.main(["check-transaction", "--json"]) == 0
    txn = json.loads(capsys.readouterr().out)
    assert txn["status"] == "pass"


def test_cli_check_all_with_skips(capsys: pytest.CaptureFixture[str]) -> None:
    code = cli.main(
        [
            "--check-all",
            "--skip-benchmark",
            "--json",
        ]
    )
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["command"] == "check-all"
    assert report["default_mode"] == "shadow"
    statuses = {item["name"]: item["status"] for item in report["checks"]}
    assert statuses["plan_objective_task_dag"] == "pass"
    assert statuses["exact_source_bindings"] == "pass"
    assert statuses["graph_index_coverage"] == "pass"
    assert statuses["proof_reconstruction"] == "pass"
    assert statuses["transaction_health"] == "pass"
    assert statuses["feature_flags"] == "pass"
    assert statuses["rollback_gates"] == "pass"
    assert statuses["guide_boundaries"] == "pass"
    assert statuses["benchmark_floors"] == "skip"
    assert code == 0
    assert report["valid"] is True


def test_cli_status(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["status", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "shadow"
    assert payload["valid"] is True


def test_cli_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    body = {
        "schema": "test/propagation-receipt@1",
        "transform": "add_argument",
        "unique_target": True,
        "reconstructed": True,
        "supported_python": True,
        "complete_frontier": True,
        "analytical_path": True,
        "language": "python",
        "roots": {"code_root": "sha256:abc"},
    }
    sealed = {**body, "receipt_id": rollout.content_identity(body)}
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps(sealed), encoding="utf-8")
    assert cli.main(["replay", "--receipt", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["valid"] is True
    assert payload["automated_mutation_authorized"] is False


def test_cli_metrics_with_skip(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["metrics", "--skip-benchmark", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["interface"] == "ChangePropagationMetrics@1"
    assert "benchmark_stages" in payload
    assert "analytical_model_split" in payload
    assert "tokens" in payload
    assert "context_bytes" in payload
    assert "fixed_point_iterations" in payload
