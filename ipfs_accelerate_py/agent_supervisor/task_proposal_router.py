"""Reusable LLM proposal routing for autonomous task-board items."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Callable, Sequence


PromptBuilder = Callable[[object, str], str]
BootstrapCallback = Callable[[], None]
DEFAULT_OPEN_TASK_STATUSES = ("to" "do", "ready")
DEFAULT_TASK_PROPOSAL_TEST_OUTPUT = "tests and fixtures needed"


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


@dataclass(frozen=True)
class TaskProposalRoutePaths:
    """Standard repo-local paths for one task-proposal route."""

    task_board_path: Path
    plan_path: Path
    artifact_dir: Path


def _repo_path(repo_root: Path, path: Path | str) -> Path:
    resolved = Path(path)
    return resolved if resolved.is_absolute() else repo_root / resolved


def build_task_proposal_route_paths(
    *,
    repo_root: Path | str,
    task_board_stem: str,
    task_board_dir: Path | str,
    plan_stem: str | None = None,
    plan_dir: Path | str | None = None,
    artifact_dir: Path | str | None = None,
    artifact_namespace: str | None = None,
    artifact_root: Path | str = "data",
    artifact_leaf: Path | str = "llm_router",
) -> TaskProposalRoutePaths:
    """Build standard repo-local paths for a task-proposal wrapper."""

    from .wrapper_utils import repo_doc_path, repo_task_board_path

    root = Path(repo_root)
    if artifact_dir is not None:
        resolved_artifact_dir = _repo_path(root, artifact_dir)
    else:
        if not artifact_namespace:
            raise ValueError("artifact_namespace is required when artifact_dir is not configured")
        resolved_artifact_dir = _repo_path(root, artifact_root) / str(artifact_namespace) / Path(artifact_leaf)
    return TaskProposalRoutePaths(
        task_board_path=repo_task_board_path(root, task_board_stem, docs_dir=task_board_dir),
        plan_path=repo_doc_path(root, f"{plan_stem or task_board_stem}.md", docs_dir=plan_dir or task_board_dir),
        artifact_dir=resolved_artifact_dir,
    )


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


def standard_task_proposal_requested_outputs(
    *domain_outputs: str,
    test_output: str = DEFAULT_TASK_PROPOSAL_TEST_OUTPUT,
) -> tuple[str, ...]:
    """Return a standard implementation-proposal output checklist."""

    outputs = ["exact files to edit"]
    outputs.extend(str(item) for item in domain_outputs if str(item))
    if test_output:
        outputs.append(str(test_output))
    outputs.extend(("validation commands", "risks or blockers"))
    return tuple(outputs)


def build_task_proposal_prompt_builder(
    *,
    intro: str,
    requested_outputs: Sequence[str],
    plan_char_limit: int = 40000,
) -> PromptBuilder:
    """Build a reusable prompt builder from project-specific wording."""

    def prompt_builder(task: object, plan_text: str) -> str:
        return build_task_proposal_prompt(
            task=task,
            plan_text=plan_text,
            intro=intro,
            requested_outputs=requested_outputs,
            plan_char_limit=plan_char_limit,
        )

    return prompt_builder


def build_task_proposal_router_cli_config(
    *,
    repo_root: Path,
    task_board_path: Path,
    task_header_prefix: str,
    plan_path: Path,
    artifact_dir: Path,
    prompt_intro: str,
    requested_outputs: Sequence[str],
    description: str,
    task_id_help: str,
    no_open_task_message: str = "No open task found.",
    task_board_option: str = "--task-board-path",
    hidden_task_board_options: Sequence[str] = (),
    include_dry_run_flag: bool = False,
    bootstrap: BootstrapCallback | None = None,
    open_statuses: Sequence[str] = DEFAULT_OPEN_TASK_STATUSES,
    plan_char_limit: int = 40000,
    provider_env: str = "IPFS_DATASETS_PY_LLM_PROVIDER",
    model_env: str = "IPFS_DATASETS_PY_LLM_MODEL",
    default_model: str = "gpt-5.3-codex-spark",
    default_max_new_tokens: int = 2048,
    default_timeout_seconds: int = 300,
) -> TaskProposalRouterCliConfig:
    """Build standard task-proposal CLI config from wrapper-specific values."""

    return TaskProposalRouterCliConfig(
        router_config=TaskProposalRouterConfig(
            repo_root=repo_root,
            task_board_path=task_board_path,
            task_header_prefix=task_header_prefix,
            plan_path=plan_path,
            artifact_dir=artifact_dir,
            prompt_builder=build_task_proposal_prompt_builder(
                intro=prompt_intro,
                requested_outputs=requested_outputs,
                plan_char_limit=plan_char_limit,
            ),
            no_open_task_message=no_open_task_message,
            open_statuses=open_statuses,
            plan_char_limit=plan_char_limit,
        ),
        description=description,
        task_id_help=task_id_help,
        task_board_option=task_board_option,
        hidden_task_board_options=hidden_task_board_options,
        include_dry_run_flag=include_dry_run_flag,
        bootstrap=bootstrap,
        provider_env=provider_env,
        model_env=model_env,
        default_model=default_model,
        default_max_new_tokens=default_max_new_tokens,
        default_timeout_seconds=default_timeout_seconds,
    )


def run_configured_task_proposal_router_cli(
    argv: list[str] | None = None,
    *,
    repo_root: Path,
    task_board_path: Path,
    task_header_prefix: str,
    plan_path: Path,
    artifact_dir: Path,
    prompt_intro: str,
    requested_outputs: Sequence[str],
    description: str,
    task_id_help: str,
    no_open_task_message: str = "No open task found.",
    task_board_option: str = "--task-board-path",
    hidden_task_board_options: Sequence[str] = (),
    include_dry_run_flag: bool = False,
    bootstrap: BootstrapCallback | None = None,
    open_statuses: Sequence[str] = DEFAULT_OPEN_TASK_STATUSES,
    plan_char_limit: int = 40000,
    provider_env: str = "IPFS_DATASETS_PY_LLM_PROVIDER",
    model_env: str = "IPFS_DATASETS_PY_LLM_MODEL",
    default_model: str = "gpt-5.3-codex-spark",
    default_max_new_tokens: int = 2048,
    default_timeout_seconds: int = 300,
) -> int:
    """Build and run the standard task-proposal router CLI from wrapper-specific values."""

    return run_task_proposal_router_cli(
        build_task_proposal_router_cli_config(
            repo_root=repo_root,
            task_board_path=task_board_path,
            task_header_prefix=task_header_prefix,
            plan_path=plan_path,
            artifact_dir=artifact_dir,
            prompt_intro=prompt_intro,
            requested_outputs=requested_outputs,
            description=description,
            task_id_help=task_id_help,
            no_open_task_message=no_open_task_message,
            task_board_option=task_board_option,
            hidden_task_board_options=hidden_task_board_options,
            include_dry_run_flag=include_dry_run_flag,
            bootstrap=bootstrap,
            open_statuses=open_statuses,
            plan_char_limit=plan_char_limit,
            provider_env=provider_env,
            model_env=model_env,
            default_model=default_model,
            default_max_new_tokens=default_max_new_tokens,
            default_timeout_seconds=default_timeout_seconds,
        ),
        argv,
    )


@dataclass(frozen=True)
class ConfiguredTaskProposalRouterRunner:
    """Project-bound runner wiring for a task-proposal router CLI."""

    config: TaskProposalRouterCliConfig

    def run(self, argv: list[str] | None = None) -> int:
        """Run the configured task-proposal router CLI."""

        return run_task_proposal_router_cli(self.config, argv)


def build_configured_task_proposal_router_runner(
    *,
    repo_root: Path,
    task_board_path: Path,
    task_header_prefix: str,
    plan_path: Path,
    artifact_dir: Path,
    prompt_intro: str,
    requested_outputs: Sequence[str],
    description: str,
    task_id_help: str,
    no_open_task_message: str = "No open task found.",
    task_board_option: str = "--task-board-path",
    hidden_task_board_options: Sequence[str] = (),
    include_dry_run_flag: bool = False,
    bootstrap: BootstrapCallback | None = None,
    open_statuses: Sequence[str] = DEFAULT_OPEN_TASK_STATUSES,
    plan_char_limit: int = 40000,
    provider_env: str = "IPFS_DATASETS_PY_LLM_PROVIDER",
    model_env: str = "IPFS_DATASETS_PY_LLM_MODEL",
    default_model: str = "gpt-5.3-codex-spark",
    default_max_new_tokens: int = 2048,
    default_timeout_seconds: int = 300,
) -> ConfiguredTaskProposalRouterRunner:
    """Build reusable task-proposal router wiring bound to project inputs."""

    return ConfiguredTaskProposalRouterRunner(
        build_task_proposal_router_cli_config(
            repo_root=repo_root,
            task_board_path=task_board_path,
            task_header_prefix=task_header_prefix,
            plan_path=plan_path,
            artifact_dir=artifact_dir,
            prompt_intro=prompt_intro,
            requested_outputs=requested_outputs,
            description=description,
            task_id_help=task_id_help,
            no_open_task_message=no_open_task_message,
            task_board_option=task_board_option,
            hidden_task_board_options=hidden_task_board_options,
            include_dry_run_flag=include_dry_run_flag,
            bootstrap=bootstrap,
            open_statuses=open_statuses,
            plan_char_limit=plan_char_limit,
            provider_env=provider_env,
            model_env=model_env,
            default_model=default_model,
            default_max_new_tokens=default_max_new_tokens,
            default_timeout_seconds=default_timeout_seconds,
        )
    )


def build_repo_task_proposal_router_runner(
    *,
    repo_root: Path | str,
    task_board_path: Path | str,
    task_header_prefix: str,
    plan_path: Path | str,
    artifact_dir: Path | str,
    prompt_intro: str,
    requested_outputs: Sequence[str],
    description: str,
    task_id_help: str,
    no_open_task_message: str = "No open task found.",
    task_board_option: str = "--task-board-path",
    hidden_task_board_options: Sequence[str] = (),
    include_dry_run_flag: bool = False,
    bootstrap: BootstrapCallback | None = None,
    runtime_package_names: Sequence[Path | str] = ("ipfs_accelerate", "ipfs_datasets"),
    runtime_external_dir: Path | str = "external",
    runtime_primary_package_names: Sequence[Path | str] | None = None,
    runtime_env_var: str = "PYTHONPATH",
    open_statuses: Sequence[str] = DEFAULT_OPEN_TASK_STATUSES,
    plan_char_limit: int = 40000,
    provider_env: str = "IPFS_DATASETS_PY_LLM_PROVIDER",
    model_env: str = "IPFS_DATASETS_PY_LLM_MODEL",
    default_model: str = "gpt-5.3-codex-spark",
    default_max_new_tokens: int = 2048,
    default_timeout_seconds: int = 300,
) -> ConfiguredTaskProposalRouterRunner:
    """Build a task-proposal runner with the standard repo runtime bootstrap."""

    resolved_repo_root = Path(repo_root)
    effective_bootstrap = bootstrap
    if effective_bootstrap is None:
        from .wrapper_utils import build_repo_runtime_environment_callbacks

        effective_bootstrap = build_repo_runtime_environment_callbacks(
            resolved_repo_root,
            runtime_package_names,
            external_dir=runtime_external_dir,
            primary_package_names=runtime_primary_package_names,
            env_var=runtime_env_var,
        ).enter
    return build_configured_task_proposal_router_runner(
        repo_root=resolved_repo_root,
        task_board_path=Path(task_board_path),
        task_header_prefix=task_header_prefix,
        plan_path=Path(plan_path),
        artifact_dir=Path(artifact_dir),
        prompt_intro=prompt_intro,
        requested_outputs=requested_outputs,
        description=description,
        task_id_help=task_id_help,
        no_open_task_message=no_open_task_message,
        task_board_option=task_board_option,
        hidden_task_board_options=hidden_task_board_options,
        include_dry_run_flag=include_dry_run_flag,
        bootstrap=effective_bootstrap,
        open_statuses=open_statuses,
        plan_char_limit=plan_char_limit,
        provider_env=provider_env,
        model_env=model_env,
        default_model=default_model,
        default_max_new_tokens=default_max_new_tokens,
        default_timeout_seconds=default_timeout_seconds,
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
        with redirect_stdout(sys.stderr):
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
