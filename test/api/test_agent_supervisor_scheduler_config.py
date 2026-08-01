from __future__ import annotations

import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_supervisor as supervisor_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    SupervisorSchedulerConfigError,
    expand_supervisor_scheduler_config_args,
    load_supervisor_scheduler_config,
    parse_args,
    supervisor_config_from_args,
)


def _write_profile(
    root: Path,
    *,
    overrides: dict[str, object] | None = None,
) -> Path:
    (root / "config").mkdir(parents=True)
    (root / "docs").mkdir()
    (root / "module-a").mkdir()
    (root / "docs" / "tasks.md").write_text("# Tasks\n", encoding="utf-8")
    (root / "docs" / "objectives.md").write_text(
        "# Objectives\n",
        encoding="utf-8",
    )
    (root / "docs" / "plan.md").write_text("# Plan\n", encoding="utf-8")
    payload: dict[str, object] = {
        "schema": (
            "ipfs_accelerate_py.agent_supervisor."
            "test.scheduler_config@1"
        ),
        "taskboard_path": "docs/tasks.md",
        "objectives_path": "docs/objectives.md",
        "task_prefix": "## TEST-",
        "board_namespace": "test-supervisor-v1",
        "merge_target_branch": "main",
        "max_lanes": 4,
        "poll_interval_seconds": 3,
        "daemon_interval_seconds": 30,
        "check_interval_seconds": 7,
        "stale_seconds": 600,
        "max_restarts": 2,
        "max_task_attempts": 3,
        "implementation_timeout_seconds": 900,
        "validation_max_workers": 2,
        "worktree_submodule_paths": ["module-a"],
        "protected_paths": [
            "docs/plan.md",
            "docs/tasks.md",
            "docs/objectives.md",
            "config/profile.json",
        ],
        "derived_refill": {"enabled_at_bootstrap": False},
        "doctor": {"mutation_authorized": False},
        "rollout": {"automatic_enabled": False},
    }
    if overrides:
        payload.update(overrides)
    path = root / "config" / "profile.json"
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def test_scheduler_config_maps_safe_defaults_and_cli_scalars_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = _write_profile(tmp_path)
    monkeypatch.setattr(supervisor_module, "REPO_ROOT", tmp_path)

    parsed = parse_args(
        [
            "--scheduler-config",
            str(profile_path.relative_to(tmp_path)),
            "--check-interval",
            "9",
            "--state-prefix",
            "operator-override",
            "--once",
        ]
    )

    assert parsed.scheduler_config == profile_path.resolve()
    assert parsed.todo_path == tmp_path / "docs" / "tasks.md"
    assert parsed.objective_path == tmp_path / "docs" / "objectives.md"
    assert parsed.task_prefix == "## TEST-"
    assert parsed.state_prefix == "operator-override"
    assert parsed.check_interval == 9
    assert parsed.daemon_interval == 30
    assert parsed.stale_seconds == 600
    assert parsed.max_restarts == 2
    assert parsed.max_task_attempts == 3
    assert parsed.implementation_timeout == 900
    assert parsed.validation_max_workers == 2
    assert parsed.merge_target_branch == "main"
    assert parsed.worktree_submodule_path == ["module-a"]
    assert parsed.implementation_protected_path == [
        "docs/plan.md",
        "docs/tasks.md",
        "docs/objectives.md",
        "config/profile.json",
    ]
    assert parsed.objective_task_janitor_enabled is False
    assert parsed.objective_reconcile_goal_completion is False
    assert parsed.implement is False
    assert parsed.objective_refill_scan is False
    assert parsed.codebase_refill_scan is False
    config = supervisor_config_from_args(parsed, repo_root=tmp_path)
    command = PortalImplementationSupervisor(config)._build_daemon_command()
    assert command[command.index("--validation-max-workers") + 1] == "2"
    assert "--implement" not in command


def test_scheduler_config_never_enables_effects_but_explicit_operator_can(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_path = _write_profile(tmp_path)
    monkeypatch.setattr(supervisor_module, "REPO_ROOT", tmp_path)

    safe = parse_args(["--scheduler-config", str(profile_path), "--once"])
    explicit = parse_args(
        ["--scheduler-config", str(profile_path), "--implement", "--once"]
    )

    assert safe.implement is False
    assert safe.objective_refill_scan is False
    assert safe.codebase_refill_scan is False
    assert explicit.implement is True


@pytest.mark.parametrize(
    ("section_name", "switch_name"),
    [
        ("derived_refill", "enabled_at_bootstrap"),
        ("doctor", "enabled_at_bootstrap"),
        ("doctor", "mutation_authorized"),
        ("doctor", "narrow_autonomous_mutation_enabled"),
        ("rollout", "automatic_enabled"),
    ],
)
def test_scheduler_config_rejects_implicit_authority_elevation(
    tmp_path: Path,
    section_name: str,
    switch_name: str,
) -> None:
    profile_path = _write_profile(
        tmp_path,
        overrides={section_name: {switch_name: True}},
    )

    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="cannot be enabled",
    ):
        load_supervisor_scheduler_config(profile_path, repo_root=tmp_path)


@pytest.mark.parametrize(
    ("field_name", "unsafe_value"),
    [
        ("taskboard_path", "../outside.md"),
        ("objectives_path", "/tmp/objectives.md"),
        ("task_prefix", "PDR-"),
        ("max_lanes", True),
        ("poll_interval_seconds", float("inf")),
        ("merge_target_branch", "../main"),
        ("protected_paths", ["docs/"]),
    ],
)
def test_scheduler_config_rejects_unsafe_or_malformed_values(
    tmp_path: Path,
    field_name: str,
    unsafe_value: object,
) -> None:
    profile_path = _write_profile(
        tmp_path,
        overrides={field_name: unsafe_value},
    )

    with pytest.raises(SupervisorSchedulerConfigError):
        load_supervisor_scheduler_config(profile_path, repo_root=tmp_path)


def test_scheduler_config_rejects_duplicate_selector_and_outside_profile(
    tmp_path: Path,
) -> None:
    profile_path = _write_profile(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text(profile_path.read_text(encoding="utf-8"), encoding="utf-8")
    try:
        with pytest.raises(
            SupervisorSchedulerConfigError,
            match="exactly once",
        ):
            expand_supervisor_scheduler_config_args(
                [
                    "--scheduler-config",
                    str(profile_path),
                    f"--scheduler-config={profile_path}",
                ],
                repo_root=tmp_path,
            )
        with pytest.raises(
            SupervisorSchedulerConfigError,
            match="inside the repository",
        ):
            load_supervisor_scheduler_config(outside, repo_root=tmp_path)
    finally:
        outside.unlink(missing_ok=True)


def test_scheduler_config_rejects_symlink_profile(tmp_path: Path) -> None:
    profile_path = _write_profile(tmp_path)
    symlink = tmp_path / "config" / "profile-link.json"
    symlink.symlink_to(profile_path.name)

    with pytest.raises(
        SupervisorSchedulerConfigError,
        match="non-symlink",
    ):
        load_supervisor_scheduler_config(symlink, repo_root=tmp_path)


def test_pdr_scheduler_profile_is_directly_consumable() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    config_path = (
        repo_root
        / "config"
        / "agent_supervisor_proof_directed_planner_doctor_scheduler.json"
    )

    profile = load_supervisor_scheduler_config(
        config_path,
        repo_root=repo_root,
    )
    expanded, selected = expand_supervisor_scheduler_config_args(
        ["--scheduler-config", str(config_path), "--once"],
        repo_root=repo_root,
    )

    assert selected == config_path.resolve()
    assert profile["task_prefix"] == "## PDR-"
    assert profile["max_lanes"] == 6
    assert "--todo-path" in expanded
    assert "--objective-path" in expanded
    assert "--no-objective-task-janitor" in expanded
    assert "--no-objective-goal-completion-reconcile" in expanded
    assert expanded[-1] == "--once"
