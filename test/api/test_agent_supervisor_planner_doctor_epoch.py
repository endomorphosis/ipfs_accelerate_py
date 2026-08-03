"""PDR-080: bounded live Planner/Doctor self-improvement epochs.

Asserts:

* lifecycle invokes the epoch controller only under explicit mode/policy;
* anchors and budgets are frozen at binding time;
* exactly one isolated challenger is used;
* every state transition is persisted;
* crash resume is idempotent;
* resource ceilings cover epochs/wall/CPU/memory/GPU/disk/tokens/cost/
  processes/storage/model calls/repairs;
* stop reasons cover safety/quality regression, unchanged residual, no
  admitted improvement, oracle/telemetry loss, rollback failure, and budget
  exhaustion;
* daemon integration never routes through test-only
  ``run_self_improvement_epoch``.
"""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.self_improvement import planner_doctor_epoch as epoch_mod
from ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_epoch import (
    DEFAULT_PROTECTED_ANCHOR_PATHS,
    MAX_CHALLENGERS_PER_EPOCH,
    MAX_EPOCHS_PER_RUN,
    MAX_GOALS_PER_EPOCH,
    MAX_TASKS_PER_EPOCH,
    PLANNER_DOCTOR_EPOCH_INTERFACE,
    PLANNER_DOCTOR_EPOCH_JOURNAL_SCHEMA,
    PLANNER_DOCTOR_EPOCH_PRODUCER_TASK_ID,
    PlannerDoctorEpochBudgets,
    PlannerDoctorEpochController,
    PlannerDoctorEpochError,
    PlannerDoctorEpochEvaluation,
    PlannerDoctorEpochMode,
    PlannerDoctorEpochPolicy,
    PlannerDoctorEpochStage,
    PlannerDoctorEpochStopReason,
    PlannerDoctorEpochUsage,
    assert_not_self_improvement_epoch_masquerade,
    build_planner_doctor_epoch_binding,
    create_isolated_challenger,
    decide_epoch_stop,
    destroy_isolated_challenger,
    freeze_planner_doctor_anchors,
    load_planner_doctor_epoch_journal,
    resume_planner_doctor_epoch,
    run_planner_doctor_epoch,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor_runner as runner_mod,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor_runner import (
    ImplementationSupervisorRunContext,
    build_supervisor_planner_doctor_epoch_callback,
    build_supervisor_planner_doctor_epoch_hooks,
)


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "ipfs_accelerate_py/agent_supervisor/self_improvement/planner_doctor_epoch.py"
)
RUNNER_PATH = (
    ROOT
    / "ipfs_accelerate_py/agent_supervisor/todo_daemon/implementation_supervisor_runner.py"
)


def _policy(
    mode: PlannerDoctorEpochMode = PlannerDoctorEpochMode.SHADOW,
    **kwargs: Any,
) -> PlannerDoctorEpochPolicy:
    return PlannerDoctorEpochPolicy(mode=mode, **kwargs)


def _binding(tmp_path: Path, *, policy: PlannerDoctorEpochPolicy | None = None):
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    # Materialize a couple of protected anchors so freeze digests are non-zero.
    for relative in DEFAULT_PROTECTED_ANCHOR_PATHS[:2]:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"anchor:{relative}\n", encoding="utf-8")
    return build_planner_doctor_epoch_binding(
        repo_root=repo,
        repository_id="repository:test",
        tree_id="tree:test-head",
        policy=policy or _policy(),
        objective_revision="objective:rev-1",
        board_revision="board:rev-1",
        capability_revision="capability:rev-1",
        protected_paths=DEFAULT_PROTECTED_ANCHOR_PATHS[:2],
    ), repo


def _eval(
    **kwargs: Any,
) -> PlannerDoctorEpochEvaluation:
    defaults = dict(
        safety_regression=False,
        quality_regression=False,
        unchanged_residual=False,
        admitted_improvement=False,
        oracle_available=True,
        telemetry_available=True,
    )
    defaults.update(kwargs)
    return PlannerDoctorEpochEvaluation(**defaults)


# ---------------------------------------------------------------------------
# Module / interface contracts
# ---------------------------------------------------------------------------


def test_module_and_interface_constants() -> None:
    assert MODULE_PATH.is_file()
    text = MODULE_PATH.read_text(encoding="utf-8")
    assert PLANNER_DOCTOR_EPOCH_INTERFACE in text
    assert "PDR-080" in text
    assert PLANNER_DOCTOR_EPOCH_PRODUCER_TASK_ID == "PDR-080"
    assert MAX_CHALLENGERS_PER_EPOCH == 1
    assert MAX_GOALS_PER_EPOCH == 8
    assert MAX_TASKS_PER_EPOCH == 24
    assert MAX_EPOCHS_PER_RUN == 8


def test_daemon_module_does_not_import_test_only_epoch_path() -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    # No import/call of the test-only ASI epoch helper (string may appear only
    # in negative assertions elsewhere; runner must not import or invoke it).
    assert "from ..self_improvement.self_improvement import" not in source
    assert "from ...self_improvement.self_improvement import" not in source
    assert "import run_self_improvement_epoch" not in source
    assert "run_self_improvement_epoch(" not in source
    assert "run_planner_doctor_epoch" in source
    assert "build_supervisor_planner_doctor_epoch_callback" in source
    # The production module also refuses masquerade re-exports.
    assert_not_self_improvement_epoch_masquerade(
        {"run_planner_doctor_epoch": object()}
    )
    with pytest.raises(PlannerDoctorEpochError):
        assert_not_self_improvement_epoch_masquerade(
            {"run_self_improvement_epoch": object()}
        )


def test_epoch_module_does_not_call_self_improvement_epoch() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    assert "run_self_improvement_epoch(" not in source
    assert "from .self_improvement import" not in source
    # Documented prohibition remains in the module docstring/contract.
    assert "run_self_improvement_epoch" in source


# ---------------------------------------------------------------------------
# Budgets / anchors / policy freezes
# ---------------------------------------------------------------------------


def test_budgets_cannot_exceed_hard_maxima() -> None:
    with pytest.raises(PlannerDoctorEpochError):
        PlannerDoctorEpochBudgets(max_goals=9)
    with pytest.raises(PlannerDoctorEpochError):
        PlannerDoctorEpochBudgets(max_challengers=2)
    budgets = PlannerDoctorEpochBudgets(max_tokens=1000)
    assert budgets.max_tokens == 1000
    assert budgets.budgets_id.startswith("sha256:")


def test_policy_off_is_disabled() -> None:
    policy = _policy(PlannerDoctorEpochMode.OFF)
    assert policy.is_enabled is False
    assert policy.to_dict()["mode"] == "off"


def test_freeze_anchors_is_content_addressed(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    relative = "docs/architecture/sample.md"
    path = repo / relative
    path.parent.mkdir(parents=True)
    path.write_text("stable\n", encoding="utf-8")
    anchors = freeze_planner_doctor_anchors(
        repo_root=repo,
        repository_id="repository:a",
        tree_id="tree:a",
        authority_policy_revision="1",
        benchmark_policy_revision="1",
        protected_paths=(relative,),
    )
    assert anchors.path_digests[relative].startswith("sha256:")
    assert anchors.verify_unmutated(repo) == ()
    path.write_text("mutated\n", encoding="utf-8")
    assert anchors.verify_unmutated(repo) == (relative,)


def test_binding_freezes_policy_budgets_and_anchors(tmp_path: Path) -> None:
    binding, _repo = _binding(tmp_path)
    payload = binding.to_dict()
    assert payload["interface"] == PLANNER_DOCTOR_EPOCH_INTERFACE
    assert payload["budgets_id"] == binding.policy.budgets.budgets_id
    assert payload["anchors_id"] == binding.anchors.anchors_id
    assert binding.epoch_id.startswith("sha256:")


# ---------------------------------------------------------------------------
# Isolated challenger
# ---------------------------------------------------------------------------


def test_single_isolated_challenger(tmp_path: Path) -> None:
    work = tmp_path / "work"
    first, root_a = create_isolated_challenger(
        work_root=work,
        epoch_id="sha256:" + ("ab" * 32),
        baseline_root="tree:base",
    )
    second, root_b = create_isolated_challenger(
        work_root=work,
        epoch_id="sha256:" + ("ab" * 32),
        baseline_root="tree:base",
    )
    assert first == second
    assert root_a == root_b
    assert (first / ".planner_doctor_challenger").is_file()
    assert destroy_isolated_challenger(first) is True
    assert not first.exists()


# ---------------------------------------------------------------------------
# Stop decisions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,expected",
    [
        ({"safety_regression": True}, PlannerDoctorEpochStopReason.SAFETY_REGRESSION),
        ({"quality_regression": True}, PlannerDoctorEpochStopReason.QUALITY_REGRESSION),
        ({"unchanged_residual": True}, PlannerDoctorEpochStopReason.UNCHANGED_RESIDUAL),
        ({"admitted_improvement": False}, PlannerDoctorEpochStopReason.NO_ADMITTED_IMPROVEMENT),
        ({"oracle_available": False}, PlannerDoctorEpochStopReason.ORACLE_LOSS),
        ({"telemetry_available": False}, PlannerDoctorEpochStopReason.TELEMETRY_LOSS),
        ({"rollback_succeeded": False}, PlannerDoctorEpochStopReason.ROLLBACK_FAILURE),
    ],
)
def test_decide_epoch_stop_reasons(kwargs: dict[str, Any], expected: PlannerDoctorEpochStopReason) -> None:
    policy = _policy()
    evaluation = _eval(**kwargs) if "admitted_improvement" in kwargs or True else _eval()
    # Rebuild with admitted_improvement default True unless testing that case.
    if "admitted_improvement" not in kwargs:
        evaluation = _eval(admitted_improvement=True, **kwargs)
    reason = decide_epoch_stop(
        policy=policy,
        evaluation=evaluation,
        usage=PlannerDoctorEpochUsage(),
    )
    assert reason is expected


def test_decide_epoch_stop_budget_exhaustion() -> None:
    policy = _policy()
    usage = PlannerDoctorEpochUsage(tokens=policy.budgets.max_tokens + 1)
    reason = decide_epoch_stop(
        policy=policy,
        evaluation=_eval(admitted_improvement=True),
        usage=usage,
    )
    assert reason is PlannerDoctorEpochStopReason.BUDGET_EXHAUSTION


# ---------------------------------------------------------------------------
# Full epoch run / persistence / resume
# ---------------------------------------------------------------------------


def test_run_epoch_under_explicit_shadow_policy(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    journal = tmp_path / "state" / "epoch-journal.json"
    work = tmp_path / "state" / "challengers"

    def provider(_binding, _baseline, _challenger):
        return _eval(
            admitted_improvement=False,
            residual_ids=("residual:live-gap",),
            evidence_ids=("evidence:live-gap",),
            detail="no improvement admitted",
        )

    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=journal,
        work_root=work,
        evaluation_provider=provider,
    )
    assert result.stop_reason is PlannerDoctorEpochStopReason.NO_ADMITTED_IMPROVEMENT
    assert result.current_stage is PlannerDoctorEpochStage.STOP
    assert result.manifest.baseline_root == "tree:test-head"
    stages = [item.stage for item in result.manifest.transitions]
    assert PlannerDoctorEpochStage.BASELINE in stages
    assert PlannerDoctorEpochStage.SHADOW in stages
    assert PlannerDoctorEpochStage.EVALUATE in stages
    assert PlannerDoctorEpochStage.STOP in stages
    # Exactly one challenger tracked.
    assert result.usage.challengers == 1
    # Journal persisted every terminal result.
    journal_payload = load_planner_doctor_epoch_journal(journal)
    assert journal_payload["schema"] == PLANNER_DOCTOR_EPOCH_JOURNAL_SCHEMA
    assert binding.epoch_id in journal_payload["epochs"]
    assert journal_payload["epochs"][binding.epoch_id]["terminal"] is True


def test_mode_off_refuses_invocation(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path, policy=_policy(PlannerDoctorEpochMode.OFF))
    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "journal.json",
        work_root=tmp_path / "work",
    )
    assert result.stop_reason is PlannerDoctorEpochStopReason.MODE_DISABLED
    assert len(result.manifest.transitions) == 1


def test_idempotent_resume_after_terminal(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    journal = tmp_path / "journal.json"
    first = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=journal,
        work_root=tmp_path / "work",
    )
    second = resume_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=journal,
        work_root=tmp_path / "work",
    )
    assert second.idempotent_replay is True
    assert second.result_id == first.result_id
    assert second.stop_reason == first.stop_reason


def test_crash_resume_from_partial_journal(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    journal = tmp_path / "journal.json"
    work = tmp_path / "work"

    # Simulate a crash after BASELINE by writing a non-terminal checkpoint.
    controller = PlannerDoctorEpochController(
        repo_root=repo,
        journal_path=journal,
        work_root=work,
        evaluation_provider=lambda *_: _eval(admitted_improvement=False),
    )
    from ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_epoch import (
        PlannerDoctorEpochTransition,
        _timestamp,
    )

    partial_transition = PlannerDoctorEpochTransition(
        stage=PlannerDoctorEpochStage.BASELINE,
        previous_stage=None,
        recorded_at=_timestamp(),
        usage=PlannerDoctorEpochUsage(epochs=1, wall_seconds=1, processes=1),
        detail="partial baseline",
        baseline_root=binding.tree_id,
    )
    controller._checkpoint(
        binding=binding,
        transitions=[partial_transition],
        usage=PlannerDoctorEpochUsage(epochs=1, wall_seconds=1, processes=1),
        baseline_root=binding.tree_id,
        challenger_root=None,
        challenger_worktree=None,
        stop_reason=None,
        evaluation=None,
        stage_spans=({"stage": "baseline", "next_stage": "propose", "elapsed_seconds": 0},),
        terminal=False,
    )
    result = resume_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=journal,
        work_root=work,
        evaluation_provider=lambda *_: _eval(admitted_improvement=False),
    )
    assert result.resumed is True
    assert result.current_stage is PlannerDoctorEpochStage.STOP
    stages = [item.stage for item in result.manifest.transitions]
    assert PlannerDoctorEpochStage.BASELINE in stages
    assert stages.count(PlannerDoctorEpochStage.BASELINE) >= 1


def test_safety_regression_stops(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
        evaluation_provider=lambda *_: _eval(
            admitted_improvement=True,
            safety_regression=True,
        ),
    )
    assert result.stop_reason is PlannerDoctorEpochStopReason.SAFETY_REGRESSION


def test_quality_regression_stops(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
        evaluation_provider=lambda *_: _eval(
            admitted_improvement=True,
            quality_regression=True,
        ),
    )
    assert result.stop_reason is PlannerDoctorEpochStopReason.QUALITY_REGRESSION


def test_oracle_and_telemetry_loss_stop(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    oracle = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "oracle.json",
        work_root=tmp_path / "w1",
        evaluation_provider=lambda *_: _eval(
            admitted_improvement=True,
            oracle_available=False,
        ),
    )
    assert oracle.stop_reason is PlannerDoctorEpochStopReason.ORACLE_LOSS

    # Different epoch identity via operator revision so journal keys differ.
    binding2, repo2 = _binding(tmp_path / "b2")
    binding2 = build_planner_doctor_epoch_binding(
        repo_root=repo2,
        repository_id="repository:test",
        tree_id="tree:test-head",
        policy=_policy(),
        objective_revision="objective:rev-1",
        board_revision="board:rev-1",
        capability_revision="capability:rev-1",
        operator_revision="operator:telemetry",
        protected_paths=DEFAULT_PROTECTED_ANCHOR_PATHS[:2],
    )
    telemetry = run_planner_doctor_epoch(
        binding=binding2,
        repo_root=repo2,
        journal_path=tmp_path / "telemetry.json",
        work_root=tmp_path / "w2",
        evaluation_provider=lambda *_: _eval(
            admitted_improvement=True,
            telemetry_available=False,
        ),
    )
    assert telemetry.stop_reason is PlannerDoctorEpochStopReason.TELEMETRY_LOSS


def test_budget_exhaustion_stops(tmp_path: Path) -> None:
    tight = PlannerDoctorEpochBudgets(max_wall_seconds=1)
    policy = _policy(budgets=tight)
    binding, repo = _binding(tmp_path, policy=policy)

    def hungry_usage(_binding, stage):
        return PlannerDoctorEpochUsage(wall_seconds=2, epochs=1, processes=1)

    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
        usage_provider=hungry_usage,
        evaluation_provider=lambda *_: _eval(admitted_improvement=True),
    )
    assert result.stop_reason is PlannerDoctorEpochStopReason.BUDGET_EXHAUSTION


def test_admitted_improvement_completes_shadow_without_promotion(
    tmp_path: Path,
) -> None:
    binding, repo = _binding(tmp_path)
    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
        evaluation_provider=lambda *_: _eval(admitted_improvement=True),
    )
    assert result.stop_reason is PlannerDoctorEpochStopReason.COMPLETED
    stages = [item.stage for item in result.manifest.transitions]
    assert PlannerDoctorEpochStage.RETAIN in stages
    assert PlannerDoctorEpochStage.RECHECK in stages
    assert PlannerDoctorEpochStage.PROMOTE not in stages


def test_every_transition_is_in_manifest(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
        evaluation_provider=lambda *_: _eval(admitted_improvement=False),
    )
    assert len(result.manifest.transitions) >= 3
    for transition in result.manifest.transitions:
        assert transition.transition_id.startswith("sha256:")
        assert transition.recorded_at


def test_usage_tracks_required_resource_dimensions(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)

    def usage_provider(_binding, stage):
        return PlannerDoctorEpochUsage(
            wall_seconds=1,
            cpu_seconds=1,
            memory_bytes=1024,
            gpu_seconds=0,
            disk_bytes=10,
            storage_bytes=10,
            tokens=5,
            cost_micros=1,
            processes=2,
            model_calls=1,
            repairs=0,
            epochs=1 if stage is PlannerDoctorEpochStage.BASELINE else 0,
            challengers=1 if stage is PlannerDoctorEpochStage.SHADOW else 0,
        )

    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
        usage_provider=usage_provider,
        evaluation_provider=lambda *_: _eval(admitted_improvement=False),
    )
    usage = result.usage.to_dict()
    for key in (
        "wall_seconds",
        "cpu_seconds",
        "memory_bytes",
        "gpu_seconds",
        "disk_bytes",
        "storage_bytes",
        "tokens",
        "cost_micros",
        "processes",
        "model_calls",
        "repairs",
        "epochs",
        "challengers",
    ):
        assert key in usage


# ---------------------------------------------------------------------------
# Lifecycle runner integration
# ---------------------------------------------------------------------------


def _run_context(tmp_path: Path, **parsed_fields: Any) -> ImplementationSupervisorRunContext:
    parsed = argparse.Namespace(
        todo_path=str(tmp_path / "todo.md"),
        task_prefix="## PDR-",
        planner_doctor_epoch_mode=parsed_fields.get("mode", "off"),
        planner_doctor_epoch_enabled=parsed_fields.get("enabled", None),
        planner_doctor_repository_id="repository:runner",
        planner_doctor_tree_id="tree:runner",
        planner_doctor_objective_revision="objective:runner",
        planner_doctor_board_revision="board:runner",
        planner_doctor_capability_revision="capability:runner",
    )
    config = argparse.Namespace(
        state_dir=tmp_path / "state",
        state_prefix="test",
    )
    return ImplementationSupervisorRunContext(
        parsed=parsed,
        config=config,
        state_path=tmp_path / "state" / "supervisor.json",
        strategy_path=tmp_path / "state" / "supervisor_strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        daemon_events_path=tmp_path / "state" / "daemon_events.jsonl",
    )


def test_lifecycle_hook_requires_explicit_mode(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    callback = build_supervisor_planner_doctor_epoch_callback(repo_root=repo)
    ctx = _run_context(tmp_path, mode="off")
    result = callback(ctx)
    assert result["invoked"] is False
    assert result["reason"] == "mode_disabled"


def test_lifecycle_hook_invokes_epoch_controller(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    for relative in DEFAULT_PROTECTED_ANCHOR_PATHS[:2]:
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x\n", encoding="utf-8")
    policy = _policy(PlannerDoctorEpochMode.SHADOW)
    callback = build_supervisor_planner_doctor_epoch_callback(
        repo_root=repo,
        policy=policy,
        evaluation_provider=lambda *_: _eval(admitted_improvement=False),
    )
    ctx = _run_context(tmp_path, mode="shadow", enabled=True)
    result = callback(ctx)
    assert result["invoked"] is True
    assert result["interface"] == PLANNER_DOCTOR_EPOCH_INTERFACE
    assert result["self_improvement_epoch_used"] is False
    assert result["stop_reason"]
    assert result["transition_count"] >= 1
    # Second call is idempotent via journal.
    again = callback(ctx)
    assert again["idempotent_replay"] is True


def test_lifecycle_hooks_builder_phases(tmp_path: Path) -> None:
    hooks = build_supervisor_planner_doctor_epoch_hooks(
        repo_root=tmp_path,
        policy=_policy(PlannerDoctorEpochMode.OBSERVE),
        before=True,
        after_once=True,
    )
    assert len(hooks) == 2
    assert {hook.phase for hook in hooks} == {"before", "after_once"}


def test_runner_source_exports_lifecycle_builders() -> None:
    assert hasattr(runner_mod, "build_supervisor_planner_doctor_epoch_callback")
    assert hasattr(runner_mod, "build_supervisor_planner_doctor_epoch_hooks")
    # Ensure the callback factory calls the production epoch entry point.
    source = inspect.getsource(runner_mod.build_supervisor_planner_doctor_epoch_callback)
    assert "run_planner_doctor_epoch" in source
    assert "run_self_improvement_epoch(" not in source
    assert "import run_self_improvement_epoch" not in source


def test_result_round_trip_dict(tmp_path: Path) -> None:
    binding, repo = _binding(tmp_path)
    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
    )
    payload = result.to_dict()
    assert payload["schema"]
    assert payload["producer_task_id"] == "PDR-080"
    # Canonical JSON must be stable.
    assert json.dumps(payload, sort_keys=True)


def test_observe_mode_skips_challenger_shadow(tmp_path: Path) -> None:
    binding, repo = _binding(
        tmp_path, policy=_policy(PlannerDoctorEpochMode.OBSERVE)
    )
    result = run_planner_doctor_epoch(
        binding=binding,
        repo_root=repo,
        journal_path=tmp_path / "j.json",
        work_root=tmp_path / "w",
        evaluation_provider=lambda *_: _eval(admitted_improvement=False),
    )
    stages = [item.stage for item in result.manifest.transitions]
    assert PlannerDoctorEpochStage.SHADOW not in stages
    assert result.usage.challengers == 0
