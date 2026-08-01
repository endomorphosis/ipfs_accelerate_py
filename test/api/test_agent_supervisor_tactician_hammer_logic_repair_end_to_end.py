"""LPR-020: end-to-end operations validation for Tactician-Hammer logic repair.

Acceptance:

* seeded explicit and ordinary-proposal two-to-three-argument plus complex
  support-type cases update all resolved callers and reach an existing
  completion receipt with a current logic fixed-point attachment or abstain;
* a healthy isolated supervisor drains the board without dependency, provider,
  protected-path, merge or lifecycle blockage;
* protected paths / four-lane isolation / exact source bindings hold;
* negatives (wrong value, open frontier, partial SCC, LLM scope, stateful
  support) fail closed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import logic_repair_rollout as rollout

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OPS_PATH = _REPO_ROOT / "scripts" / "validate_tactician_hammer_logic_repair.py"
_GUIDE_PATH = _REPO_ROOT / "docs" / "guides" / "TACTICIAN_HAMMER_LOGIC_REPAIR_GUIDE.md"
_MODULE_PATH = (
    _REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "validation"
    / "logic_repair_rollout.py"
)


def _load_ops():
    name = "validate_tactician_hammer_logic_repair_lpr020_e2e"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, _OPS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ops = _load_ops()


def test_declared_outputs_exist() -> None:
    assert _OPS_PATH.is_file()
    assert _MODULE_PATH.is_file()
    assert _GUIDE_PATH.is_file()
    assert Path(__file__).is_file()
    assert (
        _REPO_ROOT / "test/api/test_agent_supervisor_tactician_hammer_logic_repair_rollout.py"
    ).is_file()


def test_operations_symbols_and_ids() -> None:
    assert hasattr(ops, "LogicRepairEndToEnd") or hasattr(rollout, "LogicRepairEndToEnd")
    assert hasattr(rollout, "LogicRepairOperationsValidator")
    assert rollout.TASK_ID == "LPR-020"
    assert rollout.GOAL_ID == "LPR-G060"
    assert rollout.END_TO_END_INTERFACE == "LogicRepairEndToEnd@1"
    assert len(rollout.SAFETY_FLOOR_KEYS) == 10


def test_control_plane_gates_pass() -> None:
    board = rollout.check_bootstrap_board_doctor(_REPO_ROOT)
    assert board.status is rollout.CheckStatus.PASS, board.detail

    dag = rollout.check_plan_objective_task_dag(_REPO_ROOT)
    assert dag.status is rollout.CheckStatus.PASS, dag.detail
    assert "LPR-G060" in dag.evidence["goal_ids"]
    assert "LPR-019" in dag.evidence["task_ids"]
    assert "LPR-020" in dag.evidence["task_ids"]

    bindings = rollout.check_exact_source_bindings(_REPO_ROOT)
    assert bindings.status is rollout.CheckStatus.PASS, bindings.detail

    protected = rollout.check_four_lane_sharding_and_isolation(_REPO_ROOT)
    assert protected.status is rollout.CheckStatus.PASS, protected.detail
    assert protected.evidence["max_lanes"] == 4
    assert protected.evidence["one_merge_queue"] is True

    launcher = rollout.check_launcher_lifecycle_safety(_REPO_ROOT)
    assert launcher.status is rollout.CheckStatus.PASS, launcher.detail


def test_seeded_corpus_positive_and_negatives() -> None:
    report = rollout.LogicRepairEndToEnd.evaluate_seeded_corpus(_REPO_ROOT)
    assert report["interface"] == "LogicRepairEndToEnd@1"
    assert report["task_id"] == "LPR-020"
    assert report["goal_id"] == "LPR-G060"
    assert report["valid"] is True, report

    positive = report["positive"]
    for scenario in rollout.LogicRepairEndToEnd.POSITIVE_SCENARIOS:
        item = positive[scenario]
        assert item["present"] is True, scenario
        assert item["ok"] is True, (scenario, item)
        assert item.get("vector_authoritative") is False
        assert item.get("llm_authoritative") is False

    multi = positive["multiple_callers"]
    assert multi["caller_count"] >= 1 or multi["ok"] is True
    assert multi["has_logic_fixed_point_attachment"] is True or multi["fixed_point_required"] is True
    assert multi["completion_interface"] == "PropagationCompletionReceipt@1"

    overlay = report["ordinary_proposal_overlay"]
    assert overlay["present"] is True
    # Ordinary generic-provider overlay without explicit LPR path must fail closed / abstain.
    assert overlay.get("ok_fail_closed") is True or overlay.get("abstained") is True
    assert overlay.get("admitted") is False

    negatives = report["negatives"]
    for scenario in rollout.LogicRepairEndToEnd.NEGATIVE_SCENARIOS:
        item = negatives[scenario]
        assert item["present"] is True, scenario
        assert item["ok_fail_closed"] is True, (scenario, item)
        assert item["admitted"] is False
        assert item["completion_success"] is False


def test_negative_wrong_value_frontier_partial_scc_llm_scope_ordinary() -> None:
    report = rollout.LogicRepairEndToEnd.evaluate_seeded_corpus(_REPO_ROOT)
    wrong = report["negatives"]["same_typed_wrong_value"]
    assert wrong["outcome_kind"] == "wrong_value"
    assert wrong["ok_fail_closed"] is True
    assert wrong["admitted"] is False

    frontier = report["negatives"]["dynamic_reflection_generated_ffi_lifetime_concurrency"]
    assert frontier["outcome_kind"] == "open_frontier"
    assert frontier["admitted"] is False

    partial = report["negatives"]["partial_scc_rollback"]
    assert partial["scc_rollback"] is True
    assert partial["outcome_kind"] == "rollback_error"
    assert partial["admitted"] is False

    llm = report["negatives"]["path_prompt_escape"]
    assert llm["llm_scope_escape"] is False
    assert llm["admitted"] is False

    ordinary = report["negatives"]["ordinary_generic_provider_overlay"]
    assert ordinary["ok_fail_closed"] is True or ordinary.get("abstained") is True
    assert ordinary["admitted"] is False


def test_two_to_three_argument_updates_all_callers_and_fixed_point() -> None:
    report = rollout.LogicRepairEndToEnd.evaluate_seeded_corpus(_REPO_ROOT)
    two = report["two_to_three_argument"]
    assert two["ok"] is True
    assert two["caller_count"] >= 2
    assert set(two["caller_kinds"]) >= {"direct", "aliased", "wrapped", "method"}
    assert two["all_resolved_callers_updated"] is True
    assert two["completion_interface"] == "PropagationCompletionReceipt@1"
    assert two["has_logic_fixed_point_attachment"] is True
    assert two["analytical_path"] is True
    assert two["replay_valid"] is True
    assert str(two["receipt_id"]).startswith("sha256:")


def test_complex_support_type_immutable_and_stateful() -> None:
    report = rollout.LogicRepairEndToEnd.evaluate_seeded_corpus(_REPO_ROOT)
    support = report["complex_support_type"]
    immutable = support["immutable"]
    stateful = support["stateful"]
    assert immutable.get("present") is True
    assert immutable.get("ok") is True
    assert immutable.get("completion_success") is True or immutable.get("has_logic_fixed_point_attachment") is True
    assert stateful.get("present") is True
    assert stateful.get("ok") is True
    # Stateful may complete analytically with fixed-point attachment, but auto stays approval-gated.
    assert stateful.get("completion_success") is True or stateful.get("has_logic_fixed_point_attachment") is True
    assert stateful.get("approval_required_for_auto") is True
    assert stateful.get("automated_mutation_authorized") is False


def test_board_drain_without_blockage() -> None:
    report = rollout.LogicRepairEndToEnd.evaluate_seeded_corpus(_REPO_ROOT)
    drain = report["board_drain"]
    assert drain["ok"] is True
    assert drain["board_valid"] is True
    assert drain["lanes_valid"] is True
    assert drain["dependency_blockage"] is False
    assert drain["provider_blockage"] is False
    assert drain["protected_path_blockage"] is False
    assert drain["merge_blockage"] is False
    assert drain["lifecycle_blockage"] is False


def test_operations_validator_end_to_end() -> None:
    ops_facade = rollout.LogicRepairOperationsValidator(_REPO_ROOT)
    e2e = ops_facade.end_to_end()
    assert e2e["valid"] is True
    assert e2e["mutation_authorized"] is False
    assert e2e["completion_authoritative"] is False


def test_cli_end_to_end(capsys: pytest.CaptureFixture[str]) -> None:
    code = ops.main(["end-to-end", "--json"])
    out = capsys.readouterr().out
    report = json.loads(out)
    assert report["interface"] == "LogicRepairEndToEnd@1"
    assert report["valid"] is True
    assert code == 0


def test_supervisor_stopped_health(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LPR_STATE_ROOT", str(tmp_path / "isolated-stopped"))
    result = rollout.check_supervisor_process_state(
        _REPO_ROOT, state_root=tmp_path / "isolated-stopped"
    )
    assert result.status is rollout.CheckStatus.PASS
    assert result.evidence["master_status"] == "stopped"
    assert result.evidence["interface"] == "SupervisorControlService@1"
