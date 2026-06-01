from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.task_proposal_router import (
    TaskProposalRouterConfig,
    TaskProposalRouterCliConfig,
    build_task_proposal_prompt,
    run_task_proposal_router,
    run_task_proposal_router_cli,
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
