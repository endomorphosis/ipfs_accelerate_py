"""Reusable LLM proposal routing for autonomous task-board items."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence


PromptBuilder = Callable[[object, str], str]
BootstrapCallback = Callable[[], None]
DEFAULT_OPEN_TASK_STATUSES = ("to" "do", "ready")


class TaskProposalRouterError(RuntimeError):
    """Raised when a task proposal cannot be prepared."""


@dataclass(frozen=True)
class TaskProposalRouterConfig:
    """Repository-specific inputs for the reusable proposal router."""

    repo_root: Path
    task_board_path: Path
    task_header_prefix: str
    plan_path: Path
    artifact_dir: Path
    prompt_builder: PromptBuilder
    no_open_task_message: str = "No open task found."
    open_statuses: Sequence[str] = field(default_factory=lambda: DEFAULT_OPEN_TASK_STATUSES)
    plan_char_limit: int = 40000


@dataclass(frozen=True)
class TaskProposalRouterCliConfig:
    """Common CLI defaults for project-specific task proposal wrappers."""

    router_config: TaskProposalRouterConfig
    description: str
    task_id_help: str
    task_board_option: str = "--task-board-path"
    hidden_task_board_options: Sequence[str] = field(default_factory=tuple)
    include_dry_run_flag: bool = False
    bootstrap: BootstrapCallback | None = None
    provider_env: str = "IPFS_DATASETS_PY_LLM_PROVIDER"
    model_env: str = "IPFS_DATASETS_PY_LLM_MODEL"
    default_model: str = "gpt-5.3-codex-spark"
    default_max_new_tokens: int = 2048
    default_timeout_seconds: int = 300


def _task_values(task: object, name: str) -> list[str]:
    value = getattr(task, name, []) or []
    if isinstance(value, str):
        return [value] if value else []
    return [str(item) for item in value if str(item)]


def _task_value(task: object, name: str) -> str:
    return str(getattr(task, name, "") or "")


def task_metadata_lines(task: object) -> list[str]:
    """Return standard task metadata lines used by proposal prompts."""

    return [
        f"- ID: {_task_value(task, 'task_id')}",
        f"- Title: {_task_value(task, 'title')}",
        f"- Priority: {_task_value(task, 'priority')}",
        f"- Track: {_task_value(task, 'track')}",
        f"- Depends on: {', '.join(_task_values(task, 'depends_on')) or 'none'}",
        f"- Outputs: {', '.join(_task_values(task, 'outputs')) or 'none listed'}",
        f"- Validation: {'; '.join(_task_values(task, 'validation')) or 'none listed'}",
        f"- Acceptance: {_task_value(task, 'acceptance') or 'none listed'}",
    ]


def build_task_proposal_prompt(
    *,
    task: object,
    plan_text: str,
    intro: str,
    requested_outputs: Sequence[str],
    plan_char_limit: int = 40000,
) -> str:
    """Build a standard proposal prompt with project-specific framing."""

    output_lines = [f"{index}. {item}" for index, item in enumerate(requested_outputs, start=1)]
    return "\n".join(
        [
            intro.strip(),
            "",
            "Task:",
            *task_metadata_lines(task),
            "",
            "Roadmap context:",
            plan_text[: max(0, plan_char_limit)],
            "",
            "Return a concise implementation proposal with:",
            *output_lines,
            "",
        ]
    )


def select_proposal_task(
    tasks: Sequence[object],
    requested_task_id: str = "",
    *,
    open_statuses: Sequence[str] = DEFAULT_OPEN_TASK_STATUSES,
    no_open_task_message: str = "No open task found.",
) -> object:
    """Select a requested task or the first open task from a parsed task board."""

    if requested_task_id:
        for task in tasks:
            if _task_value(task, "task_id") == requested_task_id:
                return task
        raise TaskProposalRouterError(f"Unknown task id: {requested_task_id}")

    normalized_open = {str(status).strip().lower() for status in open_statuses}
    for task in tasks:
        if _task_value(task, "status").strip().lower() in normalized_open:
            return task
    raise TaskProposalRouterError(no_open_task_message)


def _artifact_relative_path(output_path: Path, repo_root: Path) -> str:
    try:
        return str(output_path.relative_to(repo_root))
    except ValueError:
        return str(output_path)


def run_task_proposal_router(
    config: TaskProposalRouterConfig,
    *,
    task_id: str = "",
    generate: bool = False,
    provider: str = "",
    model: str = "gpt-5.3-codex-spark",
    max_new_tokens: int = 2048,
    timeout_seconds: int = 300,
    allow_local_fallback: bool = False,
) -> dict[str, object]:
    """Prepare or generate an LLM implementation proposal for one task."""

    from .todo_daemon.implementation_daemon import parse_task_file
    from .todo_daemon.llm import LlmRouterInvocation, call_llm_router

    tasks = parse_task_file(config.task_board_path, config.task_header_prefix)
    selected = select_proposal_task(
        tasks,
        task_id,
        open_statuses=config.open_statuses,
        no_open_task_message=config.no_open_task_message,
    )
    plan_text = config.plan_path.read_text(encoding="utf-8")[: max(0, int(config.plan_char_limit))]
    prompt = config.prompt_builder(selected, plan_text)
    payload: dict[str, object] = {
        "task_id": _task_value(selected, "task_id"),
        "title": _task_value(selected, "title"),
        "provider": provider or None,
        "model": model,
        "prompt_chars": len(prompt),
        "generate": bool(generate),
        "llm_router_importable": True,
    }
    if not generate:
        return payload

    invocation = LlmRouterInvocation(
        repo_root=config.repo_root,
        model_name=model,
        provider=provider or None,
        allow_local_fallback=bool(allow_local_fallback),
        timeout_seconds=int(timeout_seconds),
        max_new_tokens=int(max_new_tokens),
        reject_effective_provider_name=None if allow_local_fallback else "local_hf",
    )
    proposal = call_llm_router(prompt, invocation)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    task_name = (_task_value(selected, "task_id") or "task").lower()
    output_path = config.artifact_dir / f"{task_name}-proposal.md"
    output_path.write_text(proposal, encoding="utf-8")
    payload["artifact"] = _artifact_relative_path(output_path, config.repo_root)
    return payload


def build_task_proposal_router_parser(config: TaskProposalRouterCliConfig) -> argparse.ArgumentParser:
    """Build the standard CLI parser for a project-specific proposal wrapper."""

    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--task-id", default="", help=config.task_id_help)
    parser.add_argument(
        config.task_board_option,
        dest="task_board_path",
        type=Path,
        default=config.router_config.task_board_path,
    )
    for option in config.hidden_task_board_options:
        parser.add_argument(option, dest="task_board_path", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--plan-path", type=Path, default=config.router_config.plan_path)
    parser.add_argument("--artifact-dir", type=Path, default=config.router_config.artifact_dir)
    parser.add_argument("--generate", action="store_true", help="Actually call llm_router. Default is dry-run/preflight.")
    if config.include_dry_run_flag:
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Explicit preflight mode. This is the default when --generate is not set.",
        )
    parser.add_argument("--provider", default=os.environ.get(config.provider_env, ""))
    parser.add_argument("--model", default=os.environ.get(config.model_env, config.default_model))
    parser.add_argument("--max-new-tokens", type=int, default=config.default_max_new_tokens)
    parser.add_argument("--timeout", type=int, default=config.default_timeout_seconds)
    parser.add_argument("--allow-local-fallback", action="store_true")
    return parser


def run_task_proposal_router_cli(config: TaskProposalRouterCliConfig, argv: list[str] | None = None) -> int:
    """Run the standard dry-run/generate CLI for one project-specific task board."""

    parser = build_task_proposal_router_parser(config)
    args = parser.parse_args(argv)
    if config.include_dry_run_flag and bool(getattr(args, "dry_run", False)) and args.generate:
        raise SystemExit("Choose either --generate or --dry-run, not both.")
    if config.bootstrap is not None:
        config.bootstrap()
    router_config = replace(
        config.router_config,
        task_board_path=args.task_board_path,
        plan_path=args.plan_path,
        artifact_dir=args.artifact_dir,
    )
    try:
        payload = run_task_proposal_router(
            router_config,
            task_id=args.task_id,
            generate=bool(args.generate),
            provider=args.provider,
            model=args.model,
            max_new_tokens=int(args.max_new_tokens),
            timeout_seconds=int(args.timeout),
            allow_local_fallback=bool(args.allow_local_fallback),
        )
    except TaskProposalRouterError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0
