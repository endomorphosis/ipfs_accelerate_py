"""PDR-091: protected launch profiles, lifecycle, kill switch, and runbook."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = (
    REPO_ROOT
    / "scripts"
    / "ops"
    / "agent_supervisor"
    / "proof_directed_planner_doctor.py"
)
GUIDE = REPO_ROOT / "docs" / "guides" / "PROOF_DIRECTED_PLANNER_DOCTOR_GUIDE.md"


def _load_ops():
    name = "proof_directed_planner_doctor_ops"
    spec = importlib.util.spec_from_file_location(name, CLI)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    # dataclasses require the module to be registered before class bodies run
    sys.modules[name] = mod
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    spec.loader.exec_module(mod)
    return mod


ops = _load_ops()


def _run_cli(*args: str, env: dict | None = None) -> subprocess.CompletedProcess[str]:
    command_env = {
        **dict(os.environ),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "IPFS_ACCEL_SKIP_CORE": "1",
        "PYTHONPATH": str(REPO_ROOT)
        + (":" + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    if env:
        command_env.update(env)
    return subprocess.run(
        [sys.executable, str(CLI), *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env=command_env,
        check=False,
    )


# ---------------------------------------------------------------------------
# Interfaces / defaults
# ---------------------------------------------------------------------------


def test_interface_and_seed_defaults() -> None:
    assert ops.PLANNER_DOCTOR_OPERATIONS_INTERFACE == "PlannerDoctorOperations@1"
    assert ops.PRODUCER_TASK_ID == "PDR-091"
    assert ops.GOAL_ID == "PDR-G100"
    assert ops.MAX_SEED_LANES == 6
    assert ops.BOARD_NAMESPACE == "agent-supervisor-proof-directed-planner-doctor-v1"

    profile = ops.default_launch_profile()
    assert profile.doctor_mode == "report_only"
    assert profile.planner_mode == "shadow"
    assert profile.rollout_mode == "shadow"
    assert profile.automatic_enabled is False
    assert profile.doctor_mutation_authorized is False
    assert profile.doctor_enabled is False
    assert profile.refill_enabled is False
    assert profile.max_lanes == 6
    payload = profile.to_dict()
    assert payload["interface"] == "PlannerDoctorOperations@1"
    assert payload["max_seed_lanes"] == 6


def test_launch_profile_from_sealed_scheduler() -> None:
    profile = ops.launch_profile_from_scheduler(REPO_ROOT)
    assert profile.max_lanes <= ops.MAX_SEED_LANES
    assert profile.doctor_mode == "report_only"
    assert profile.planner_mode == "shadow"
    assert profile.automatic_enabled is False
    assert profile.refill_enabled is False
    assert profile.doctor_mutation_authorized is False
    assert profile.board_namespace == ops.BOARD_NAMESPACE
    assert "resource_hints" in profile.to_dict()
    assert profile.resource_hints  # telemetry declared in scheduler


def test_seed_lanes_reject_above_six() -> None:
    with pytest.raises(ops.PlannerDoctorOperationsError):
        ops.LaunchProfile(max_lanes=7)
    with pytest.raises(ops.PlannerDoctorOperationsError):
        ops.launch_profile_from_scheduler(REPO_ROOT, requested_lanes=7)


def test_lifecycle_commands_are_closed() -> None:
    expected = {
        "validate",
        "plan",
        "start",
        "status",
        "stop",
        "restart",
        "pause",
        "drain",
        "benchmark",
        "promote",
        "rollback",
        "kill-switch",
        "kill-switch-clear",
        "recipe",
        "deposit-receipt",
    }
    assert set(ops.LIFECYCLE_COMMANDS) == expected
    help_text = ops.build_parser().format_help()
    for name in expected:
        assert name in help_text


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_launch_on_repo_with_allow_dirty(tmp_path: Path) -> None:
    state_dir = tmp_path / "ops-state"
    report = ops.validate_launch(
        REPO_ROOT,
        state_dir,
        require_clean=False,
    )
    assert report["interface"] == "PlannerDoctorOperations@1"
    assert report["checks"]["defaults"]["ok"] is True
    assert report["checks"]["protected_anchors"]["ok"] is True
    assert report["checks"]["board_objective_dag"]["ok"] is True
    assert report["checks"]["board_objective_dag"]["task_dag_acyclic"] is True
    assert report["checks"]["board_objective_dag"]["goal_dag_acyclic"] is True
    assert report["checks"]["lanes"]["max_seed_lanes"] == 6
    assert report["checks"]["isolation"]["ok"] is True
    assert report["checks"]["capabilities"]["ok"] is True
    gates = report["checks"]["feature_gates"]
    assert gates["all_privileged_off"] is True
    assert gates["automatic"]["unlocked"] is False
    assert gates["doctor_mutation"]["unlocked"] is False
    assert gates["refill"]["unlocked"] is False
    # Dirty checkout may fail clean_target when require_clean=True; with
    # allow-dirty path, ok depends on other checks.
    assert "clean_target" in report["checks"]
    assert report["ok"] is True


def test_validate_rejects_when_protected_anchor_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    profile = ops.default_launch_profile()
    # Point taskboard to a missing path via a custom profile.
    broken = ops.LaunchProfile(
        protected_paths=("docs/architecture/DOES_NOT_EXIST.md",),
        taskboard_path=profile.taskboard_path,
        objectives_path=profile.objectives_path,
    )
    state_dir = tmp_path / "state"
    report = ops.validate_launch(
        REPO_ROOT,
        state_dir,
        broken,
        require_clean=False,
    )
    assert report["ok"] is False
    assert "protected_anchors" in report["failed_checks"]


def test_board_dag_detects_cycle() -> None:
    ready, cycles, acyclic = ops._topo_ready(
        {"A": ["B"], "B": ["C"], "C": ["A"]}
    )
    assert acyclic is False
    assert set(cycles) == {"A", "B", "C"}
    ready2, cycles2, acyclic2 = ops._topo_ready(
        {"A": [], "B": ["A"], "C": ["B"]}
    )
    assert acyclic2 is True
    assert cycles2 == []
    assert "A" in ready2


# ---------------------------------------------------------------------------
# Lifecycle / idempotency / fencing
# ---------------------------------------------------------------------------


def test_plan_start_status_stop_restart_are_idempotent_and_fenced(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "ops"
    plan1 = ops.plan_launch(
        REPO_ROOT, state_dir, require_clean=False, lanes=2
    )
    assert plan1["ok"] is True
    assert plan1["phase"] == "planned"
    assert plan1["fence_token"]
    plan2 = ops.plan_launch(
        REPO_ROOT, state_dir, require_clean=False, lanes=2
    )
    assert plan2["ok"] is True
    assert "idempotent" in plan2["reason_codes"]

    start1 = ops.start_run(
        REPO_ROOT, state_dir, require_clean=False, lanes=2
    )
    assert start1["ok"] is True
    assert start1["phase"] == "running"
    assert start1["dispatch_allowed"] is True
    start2 = ops.start_run(
        REPO_ROOT, state_dir, require_clean=False, lanes=2
    )
    assert "idempotent" in start2["reason_codes"]

    status = ops.status_run(state_dir)
    assert status["ok"] is True
    assert status["health"]["phase"] == "running"
    assert status["health"]["lanes"] == 2
    assert status["feature_gates"]["all_privileged_off"] is True

    stop1 = ops.stop_run(state_dir)
    assert stop1["ok"] is True
    assert stop1["dispatch_allowed"] is False
    stop2 = ops.stop_run(state_dir)
    assert "idempotent" in stop2["reason_codes"]

    restarted = ops.restart_run(
        REPO_ROOT, state_dir, require_clean=False, lanes=2
    )
    assert restarted["ok"] is True
    assert restarted["phase"] == "running"


def test_pause_and_drain(tmp_path: Path) -> None:
    state_dir = tmp_path / "ops"
    ops.plan_launch(REPO_ROOT, state_dir, require_clean=False, lanes=1)
    ops.start_run(REPO_ROOT, state_dir, require_clean=False, lanes=1)
    paused = ops.pause_run(state_dir)
    assert paused["ok"] is True
    assert paused["phase"] == "paused"
    assert paused["dispatch_allowed"] is False
    assert "idempotent" in ops.pause_run(state_dir)["reason_codes"]

    # Resume via start, then drain.
    ops.start_run(REPO_ROOT, state_dir, require_clean=False, lanes=1)
    drained = ops.drain_run(state_dir)
    assert drained["ok"] is True
    assert drained["phase"] == "draining"
    assert drained["dispatch_allowed"] is False
    assert "future_dispatch_cancelled" in drained["reason_codes"]


def test_promote_one_stage_and_rollback(tmp_path: Path) -> None:
    state_dir = tmp_path / "ops"
    ops.plan_launch(REPO_ROOT, state_dir, require_clean=False)
    state = ops.load_state(state_dir)
    assert state.rollout_mode == "shadow"

    # shadow -> assist
    promoted = ops.promote_one_stage(state_dir)
    assert promoted["ok"] is True
    assert promoted["to_mode"] == "assist"
    # assist -> canary
    promoted2 = ops.promote_one_stage(state_dir)
    assert promoted2["to_mode"] == "canary"
    # canary -> automatic blocked without receipt + elevated profile
    blocked = ops.promote_one_stage(state_dir)
    assert blocked["ok"] is False
    assert "automatic_requires_prerequisite_receipt" in blocked["reason_codes"]

    rolled = ops.rollback_stage(state_dir)
    assert rolled["ok"] is True
    assert rolled["to_mode"] == "assist"
    rolled2 = ops.rollback_stage(state_dir, to_mode="shadow")
    assert rolled2["ok"] is True
    assert rolled2["to_mode"] == "shadow"
    # Cannot elevate via rollback
    bad = ops.rollback_stage(state_dir, to_mode="canary")
    assert bad["ok"] is False
    assert "rollback_cannot_elevate" in bad["reason_codes"]


# ---------------------------------------------------------------------------
# Kill switch
# ---------------------------------------------------------------------------


def test_kill_switch_forces_report_only_cancels_dispatch_blocks_promotion(
    tmp_path: Path,
) -> None:
    state_dir = tmp_path / "ops"
    ops.plan_launch(REPO_ROOT, state_dir, require_clean=False, lanes=2)
    ops.start_run(REPO_ROOT, state_dir, require_clean=False, lanes=2)
    ops.promote_one_stage(state_dir)  # shadow -> assist

    engaged = ops.engage_kill_switch(state_dir, reason="test_engage")
    assert engaged["ok"] is True
    assert engaged["kill_switch_engaged"] is True
    assert engaged["dispatch_allowed"] is False
    assert engaged["promotion_blocked"] is True
    assert engaged["doctor_mode"] == "report_only"
    assert "report_only_forced" in engaged["reason_codes"]
    assert "dispatch_cancelled" in engaged["reason_codes"]
    assert "promotion_blocked" in engaged["reason_codes"]
    # Idempotent re-engage
    assert "idempotent" in ops.engage_kill_switch(state_dir)["reason_codes"]

    status = ops.status_run(state_dir)
    assert status["effective_doctor_mode"] == "report_only"
    assert status["health"]["dispatch_allowed"] is False
    assert status["health"]["promotion_blocked"] is True

    # Future start blocked
    started = ops.start_run(REPO_ROOT, state_dir, require_clean=False)
    assert started["ok"] is False
    assert "kill_switch_engaged" in started["reason_codes"]

    # Promotion blocked
    promoted = ops.promote_one_stage(state_dir)
    assert promoted["ok"] is False
    assert "promotion_blocked" in promoted["reason_codes"] or (
        "kill_switch_engaged" in promoted["reason_codes"]
    )

    # Clear requires operator ack
    denied = ops.clear_kill_switch(state_dir, operator_ack=False)
    assert denied["ok"] is False
    cleared = ops.clear_kill_switch(state_dir, operator_ack=True)
    assert cleared["ok"] is True
    assert "remains_report_only" in cleared["reason_codes"]
    state = ops.load_state(state_dir)
    assert state.kill_switch_engaged is False
    assert state.doctor_mode == "report_only"
    assert state.dispatch_allowed is False


def test_restart_preserves_kill_switch(tmp_path: Path) -> None:
    state_dir = tmp_path / "ops"
    ops.plan_launch(REPO_ROOT, state_dir, require_clean=False)
    ops.start_run(REPO_ROOT, state_dir, require_clean=False)
    ops.engage_kill_switch(state_dir)
    restarted = ops.restart_run(REPO_ROOT, state_dir, require_clean=False)
    assert restarted["ok"] is False
    assert "kill_switch_engaged" in restarted["reason_codes"]


# ---------------------------------------------------------------------------
# Feature gates / receipts
# ---------------------------------------------------------------------------


def test_privileged_features_off_until_receipts(tmp_path: Path) -> None:
    state_dir = tmp_path / "ops"
    state_dir.mkdir(parents=True)
    profile = ops.default_launch_profile()
    gates = ops.feature_gate_status(state_dir, profile, completed_task_ids=())
    assert gates["refill"]["unlocked"] is False
    assert gates["doctor_mutation"]["unlocked"] is False
    assert gates["automatic"]["unlocked"] is False

    # Board complete alone is insufficient.
    gates2 = ops.feature_gate_status(
        state_dir,
        profile,
        completed_task_ids=("PDR-081", "PDR-053", "PDR-090"),
    )
    assert gates2["refill"]["unlocked"] is False
    assert gates2["doctor_mutation"]["unlocked"] is False

    # Receipt alone is insufficient without board completion.
    ops.deposit_prerequisite_receipt(
        state_dir,
        "refill",
        task_id="PDR-081",
        evidence_id="evidence:refill-test@1",
    )
    gates3 = ops.feature_gate_status(state_dir, profile, completed_task_ids=())
    assert gates3["refill"]["receipt_present"] is True
    assert gates3["refill"]["unlocked"] is False

    # Both present unlocks refill (not automatic).
    gates4 = ops.feature_gate_status(
        state_dir, profile, completed_task_ids=("PDR-081",)
    )
    assert gates4["refill"]["unlocked"] is True
    assert gates4["automatic"]["unlocked"] is False

    # Wrong task id for feature fails closed.
    with pytest.raises(ops.PlannerDoctorOperationsError):
        ops.deposit_prerequisite_receipt(
            state_dir,
            "refill",
            task_id="PDR-001",
            evidence_id="evidence:bad@1",
        )


def test_benchmark_gate_does_not_elevate(tmp_path: Path) -> None:
    state_dir = tmp_path / "ops"
    ops.plan_launch(REPO_ROOT, state_dir, require_clean=False)
    report = ops.run_benchmark_gate(state_dir)
    assert report["ok"] is True
    assert report["benchmark"]["synthetic_evidence_may_promote"] is False
    assert report["benchmark"]["automatic_still_off"] is True
    assert report["benchmark"]["configured_maximum_lanes"] <= 6


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_help_and_recipe() -> None:
    help_result = _run_cli("--help")
    assert help_result.returncode == 0
    assert "validate" in help_result.stdout
    assert "kill-switch" in help_result.stdout
    assert "PlannerDoctorOperations@1" in help_result.stdout

    recipe = _run_cli("recipe")
    assert recipe.returncode == 0
    payload = json.loads(recipe.stdout)
    assert payload["interface"] == "PlannerDoctorOperations@1"
    assert payload["max_seed_lanes"] == 6
    assert payload["defaults"]["doctor_mode"] == "report_only"
    assert payload["defaults"]["automatic_enabled"] is False


def test_cli_validate_and_status_with_state_dir(tmp_path: Path) -> None:
    state_dir = tmp_path / "cli-state"
    result = _run_cli(
        "--state-dir",
        str(state_dir),
        "--allow-dirty",
        "validate",
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["checks"]["defaults"]["doctor_mode"] == "report_only"

    plan = _run_cli(
        "--state-dir",
        str(state_dir),
        "--allow-dirty",
        "--lanes",
        "2",
        "plan",
    )
    assert plan.returncode == 0, plan.stderr
    kill = _run_cli("--state-dir", str(state_dir), "kill-switch")
    assert kill.returncode == 0
    kill_payload = json.loads(kill.stdout)
    assert kill_payload["kill_switch_engaged"] is True

    status = _run_cli("--state-dir", str(state_dir), "status")
    assert status.returncode == 0
    status_payload = json.loads(status.stdout)
    assert status_payload["effective_doctor_mode"] == "report_only"
    assert status_payload["health"]["promotion_blocked"] is True


def test_cli_missing_command_is_usage_error() -> None:
    result = _run_cli()
    assert result.returncode == 2


# ---------------------------------------------------------------------------
# Runbook / guide
# ---------------------------------------------------------------------------


def test_guide_covers_runbook_topics() -> None:
    assert GUIDE.is_file()
    text = GUIDE.read_text(encoding="utf-8")
    required = (
        "report-only",
        "shadow",
        "kill switch",
        "capability degradation",
        "stale state",
        "rollback",
        "quarantine",
        "held-out",
        "holdout",
        "recovery",
        "protected anchor",
        "six seed",
        "automatic",
        "refill",
        "Doctor mutation",
        "idempotent",
        "PlannerDoctorOperations@1",
        "proof_directed_planner_doctor.py",
    )
    lowered = text.lower()
    for term in required:
        assert term.lower() in lowered, f"guide missing topic: {term}"


def test_ops_script_does_not_write_protected_anchors(tmp_path: Path) -> None:
    """Lifecycle ops must only touch isolated state_dir, not anchors."""

    state_dir = tmp_path / "ops"
    before = {
        rel: (REPO_ROOT / rel).stat().st_mtime_ns
        for rel in ops.SEED_PROTECTED_ANCHORS
        if (REPO_ROOT / rel).is_file()
    }
    ops.plan_launch(REPO_ROOT, state_dir, require_clean=False)
    ops.start_run(REPO_ROOT, state_dir, require_clean=False)
    ops.engage_kill_switch(state_dir)
    ops.promote_one_stage(state_dir)
    ops.rollback_stage(state_dir)
    ops.stop_run(state_dir)
    after = {
        rel: (REPO_ROOT / rel).stat().st_mtime_ns
        for rel in before
    }
    assert before == after
    assert (state_dir / "operations_state.json").is_file()


def test_recipe_lists_all_lifecycle_effects() -> None:
    recipe = ops.launch_recipe()
    for cmd in (
        "validate",
        "plan",
        "start",
        "status",
        "stop",
        "restart",
        "pause",
        "drain",
        "benchmark",
        "promote",
        "rollback",
        "kill-switch",
    ):
        assert cmd in recipe["commands"]
