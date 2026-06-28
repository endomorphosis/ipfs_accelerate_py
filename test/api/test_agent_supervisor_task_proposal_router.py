from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    TaskProposalRouterConfig,
    TaskProposalRouterCliConfig,
    TaskProposalRoutePaths,
    build_task_proposal_route_paths,
    build_task_proposal_prompt_builder,
    build_task_proposal_prompt,
    build_task_proposal_router_cli_config,
    run_task_proposal_router,
    run_task_proposal_router_cli,
)


def test_task_proposal_route_paths_resolve_repo_local_defaults(tmp_path: Path):
    paths = build_task_proposal_route_paths(
        repo_root=tmp_path,
        task_board_stem="roadmap",
        task_board_dir="docs",
        artifact_namespace="agent_tracks",
    )

    assert isinstance(paths, TaskProposalRoutePaths)
    assert paths.task_board_path == tmp_path / "docs" / "roadmap.todo.md"
    assert paths.plan_path == tmp_path / "docs" / "roadmap.md"
    assert paths.artifact_dir == tmp_path / "data" / "agent_tracks" / "llm_router"


def test_task_proposal_route_paths_accept_plan_and_artifact_overrides(tmp_path: Path):
    artifact_dir = tmp_path / "artifacts"

    paths = build_task_proposal_route_paths(
        repo_root=tmp_path,
        task_board_stem="tasks",
        task_board_dir="docs",
        plan_stem="plan",
        plan_dir="plans",
        artifact_dir=artifact_dir,
    )

    assert paths.task_board_path == tmp_path / "docs" / "tasks.todo.md"
    assert paths.plan_path == tmp_path / "plans" / "plan.md"
    assert paths.artifact_dir == artifact_dir


def test_task_proposal_route_paths_require_artifact_namespace_without_artifact_dir(tmp_path: Path):
    with pytest.raises(ValueError, match="artifact_namespace is required"):
        build_task_proposal_route_paths(
            repo_root=tmp_path,
            task_board_stem="roadmap",
            task_board_dir="docs",
        )


def test_task_proposal_router_dry_run_selects_open_task(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    board = repo / "tasks.todo.md"
    plan = repo / "plan.md"
    artifacts = repo / "artifacts"
    board.write_text(
        "\n".join(
            [
                "# Tasks",
                "",
                "## TST-001 Completed task",
                "- Status: completed",
                "- Priority: P1",
                "- Track: ops",
                "- Depends on:",
                "- Outputs: completed.py",
                "- Validation: pytest completed",
                "- Acceptance: Already done.",
                "",
                "## TST-002 Ready task",
                "- Status: ready",
                "- Priority: P0",
                "- Track: runtime",
                "- Depends on: TST-001",
                "- Outputs: runtime.py, tests/test_runtime.py",
                "- Validation: pytest tests/test_runtime.py",
                "- Acceptance: Runtime proposal is generated.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan.write_text("Reusable router roadmap context.", encoding="utf-8")

    selected_ids: list[str] = []

    def prompt_builder(task: object, plan_text: str) -> str:
        selected_ids.append(str(getattr(task, "task_id", "")))
        return build_task_proposal_prompt(
            task=task,
            plan_text=plan_text,
            intro="You are helping implement reusable supervisor work.",
            requested_outputs=(
                "exact files to edit",
                "tests and fixtures needed",
                "validation commands",
            ),
        )

    payload = run_task_proposal_router(
        TaskProposalRouterConfig(
            repo_root=repo,
            task_board_path=board,
            task_header_prefix="## TST-",
            plan_path=plan,
            artifact_dir=artifacts,
            prompt_builder=prompt_builder,
            no_open_task_message="No reusable task found.",
        ),
        provider="test-provider",
        model="test-model",
    )

    assert selected_ids == ["TST-002"]
    assert payload["task_id"] == "TST-002"
    assert payload["title"] == "Ready task"
    assert payload["provider"] == "test-provider"
    assert payload["model"] == "test-model"
    assert payload["generate"] is False
    assert payload["llm_router_importable"] is True
    assert payload["prompt_chars"] > 100
    assert not artifacts.exists()


def test_task_proposal_router_cli_reuses_common_dry_run_flow(tmp_path: Path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    board = repo / "tasks.todo.md"
    plan = repo / "plan.md"
    artifacts = repo / "artifacts"
    board.write_text(
        "\n".join(
            [
                "# Tasks",
                "",
                "## TST-001 Ready task",
                "- Status: ready",
                "- Priority: P1",
                "- Track: runtime",
                "- Depends on:",
                "- Outputs: runtime.py",
                "- Validation: pytest tests/test_runtime.py",
                "- Acceptance: Runtime proposal is generated.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan.write_text("Reusable CLI router roadmap context.", encoding="utf-8")
    bootstrap_calls: list[str] = []

    def prompt_builder(task: object, plan_text: str) -> str:
        return build_task_proposal_prompt(
            task=task,
            plan_text=plan_text,
            intro="You are helping implement reusable supervisor work.",
            requested_outputs=("exact files to edit", "validation commands"),
        )

    result = run_task_proposal_router_cli(
        TaskProposalRouterCliConfig(
            router_config=TaskProposalRouterConfig(
                repo_root=repo,
                task_board_path=board,
                task_header_prefix="## TST-",
                plan_path=plan,
                artifact_dir=artifacts,
                prompt_builder=prompt_builder,
            ),
            description="Generate a reusable proposal.",
            task_id_help="Specific test task.",
            task_board_option="--todo-path",
            hidden_task_board_options=("--task-board-path",),
            bootstrap=lambda: bootstrap_calls.append("called"),
        ),
        ["--task-id", "TST-001", "--provider", "cli-provider", "--model", "cli-model"],
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert bootstrap_calls == ["called"]
    assert payload["task_id"] == "TST-001"
    assert payload["provider"] == "cli-provider"
    assert payload["model"] == "cli-model"
    assert payload["generate"] is False
    assert not artifacts.exists()


def test_task_proposal_router_cli_config_factory_reuses_prompt_builder(tmp_path: Path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    board = repo / "tasks.todo.md"
    plan = repo / "plan.md"
    artifacts = repo / "artifacts"
    board.write_text(
        "\n".join(
            [
                "# Tasks",
                "",
                "## TST-001 Ready task",
                "- Status: ready",
                "- Priority: P1",
                "- Track: runtime",
                "- Depends on:",
                "- Outputs: runtime.py",
                "- Validation: pytest tests/test_runtime.py",
                "- Acceptance: Runtime proposal is generated.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    plan.write_text("Reusable factory router roadmap context.", encoding="utf-8")
    bootstrap_calls: list[str] = []

    config = build_task_proposal_router_cli_config(
        repo_root=repo,
        task_board_path=board,
        task_header_prefix="## TST-",
        plan_path=plan,
        artifact_dir=artifacts,
        prompt_intro="You are helping implement reusable supervisor work.",
        requested_outputs=("exact files to edit", "validation commands"),
        description="Generate a reusable proposal.",
        task_id_help="Specific test task.",
        task_board_option="--todo-path",
        hidden_task_board_options=("--task-board-path",),
        bootstrap=lambda: bootstrap_calls.append("called"),
    )

    result = run_task_proposal_router_cli(
        config,
        ["--task-id", "TST-001", "--provider", "factory-provider", "--model", "factory-model"],
    )

    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert bootstrap_calls == ["called"]
    assert payload["task_id"] == "TST-001"
    assert payload["provider"] == "factory-provider"
    assert payload["model"] == "factory-model"
    assert payload["generate"] is False
    assert not artifacts.exists()


def test_task_proposal_prompt_builder_applies_plan_limit():
    class Task:
        task_id = "TST-001"
        title = "Ready task"
        priority = "P1"
        track = "runtime"
        depends_on: list[str] = []
        outputs: list[str] = []
        validation: list[str] = []
        acceptance = "Runtime proposal is generated."

    prompt_builder = build_task_proposal_prompt_builder(
        intro="You are helping implement reusable supervisor work.",
        requested_outputs=("validation commands",),
        plan_char_limit=4,
    )

    prompt = prompt_builder(Task(), "abcdef")

    assert "abcd" in prompt
    assert "abcdef" not in prompt
    assert "validation commands" in prompt
