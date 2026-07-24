"""Reusable LLM proposal routing for autonomous task-board items."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from contextlib import redirect_stdout
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from .context_compiler import (
    ContextCompileResult,
    ContextCompiler,
    build_text_context_references,
    render_context_capsule,
)
from .context_contracts import ContextBudget
from .plan_evaluator import (
    ANALYSIS_PROPOSAL_JSON_SCHEMA,
    PLAN_BRANCH_JSON_SCHEMA,
    AnalysisProposal,
    AnalysisProposalEvaluation,
    PlanBranch,
    PlanBranchValidationError,
    evaluate_analysis_proposals,
)


PromptBuilder = Callable[[object, str], str]
BootstrapCallback = Callable[[], None]
DEFAULT_OPEN_TASK_STATUSES = ("to" "do", "ready")
DEFAULT_TASK_PROPOSAL_TEST_OUTPUT = "tests and fixtures needed"
TASK_IMPLEMENTATION_PROPOSAL_SCHEMA = (
    "ipfs_accelerate_py/agent-supervisor/task-implementation-proposal@1"
)
TASK_PROPOSAL_MAX_RESPONSE_BYTES = 256_000
TASK_PROPOSAL_MAX_JSON_DEPTH = 12
TASK_PROPOSAL_MAX_FILES = 256
TASK_PROPOSAL_OPERATIONS = frozenset({"add", "modify", "delete", "rename"})
DEFAULT_PLANNING_CONTEXT_INPUT_TOKENS = 12_288
DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE = 512

TASK_IMPLEMENTATION_PROPOSAL_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "schema",
        "proposal_version",
        "task_id",
        "repository_tree_id",
        "context_id",
        "files",
        "validation_plan",
        "risks",
        "authority_claims",
    ],
    "properties": {
        "schema": {"const": TASK_IMPLEMENTATION_PROPOSAL_SCHEMA},
        "proposal_version": {"const": "1"},
        "task_id": {"type": "string", "minLength": 1},
        "repository_tree_id": {"type": "string", "minLength": 1},
        "context_id": {"type": "string", "minLength": 1},
        "files": {
            "type": "array",
            "minItems": 1,
            "maxItems": TASK_PROPOSAL_MAX_FILES,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["path", "operation", "rationale_references"],
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "operation": {
                        "enum": sorted(TASK_PROPOSAL_OPERATIONS)
                    },
                    "rationale_references": {
                        "type": "array",
                        "minItems": 1,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        "validation_plan": {
            "type": "array",
            "items": {"type": "string", "minLength": 1},
        },
        "risks": {
            "type": "array",
            "minItems": 1,
            "items": {"type": "string", "minLength": 1},
        },
        "authority_claims": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "allowed_paths",
                "validation_commands_only",
                "proof_authoritative",
                "completion_authoritative",
            ],
            "properties": {
                "allowed_paths": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                },
                "validation_commands_only": {"const": True},
                "proof_authoritative": {"const": False},
                "completion_authoritative": {"const": False},
            },
        },
    },
}


class TaskProposalRouterError(RuntimeError):
    """Raised when a task proposal cannot be prepared."""

    def __init__(self, message: str, *, reason_code: str = "proposal_router_error") -> None:
        super().__init__(message)
        self.reason_code = str(reason_code or "proposal_router_error")


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
    context_max_input_tokens: int | None = None
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE
    provider_context_window: int | None = None
    provider_max_input_tokens: int | None = None
    context_tokenizer: Any = field(default=None, repr=False, compare=False)


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


@dataclass(frozen=True)
class TaskProposalRouteSpec:
    """Project-specific task-proposal route values without repo-root binding."""

    task_board_stem: str
    task_board_dir: Path | str
    artifact_namespace: str
    task_header_prefix: str
    prompt_intro: str
    description: str
    task_id_help: str
    domain_outputs: Sequence[str] = field(default_factory=tuple)
    test_output: str = DEFAULT_TASK_PROPOSAL_TEST_OUTPUT
    requested_outputs: Sequence[str] | None = None
    no_open_task_message: str = "No open task found."
    task_board_option: str | None = None
    hidden_task_board_options: Sequence[str] = field(default_factory=tuple)
    hidden_standard_task_board_option: bool = False
    include_dry_run_flag: bool = False
    plan_stem: str | None = None
    plan_dir: Path | str | None = None
    artifact_dir: Path | str | None = None
    artifact_root: Path | str = "data"
    artifact_leaf: Path | str = "llm_router"
    bootstrap: BootstrapCallback | None = None
    runtime_package_names: Sequence[Path | str] = field(
        default_factory=lambda: ("ipfs_accelerate", "ipfs_datasets")
    )
    runtime_external_dir: Path | str = "external"
    runtime_primary_package_names: Sequence[Path | str] | None = None
    runtime_env_var: str = "PYTHONPATH"
    open_statuses: Sequence[str] = field(default_factory=lambda: DEFAULT_OPEN_TASK_STATUSES)
    plan_char_limit: int = 40000
    context_max_input_tokens: int | None = None
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE
    provider_context_window: int | None = None
    provider_max_input_tokens: int | None = None
    context_tokenizer: Any = field(default=None, repr=False, compare=False)
    provider_env: str = "IPFS_DATASETS_PY_LLM_PROVIDER"
    model_env: str = "IPFS_DATASETS_PY_LLM_MODEL"
    default_model: str = "gpt-5.3-codex-spark"
    default_max_new_tokens: int = 2048
    default_timeout_seconds: int = 300


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
    """Build project-specific framing without selecting roadmap evidence.

    ``plan_char_limit`` is retained as a compatibility hint for router
    configuration.  The live router translates it into a token budget; this
    formatter deliberately keeps the supplied artifact complete so it never
    creates an unaudited partial roadmap.
    """

    output_lines = [f"{index}. {item}" for index, item in enumerate(requested_outputs, start=1)]
    return "\n".join(
        [
            intro.strip(),
            "",
            "Task:",
            *task_metadata_lines(task),
            "",
            "Roadmap context:",
            plan_text,
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
    context_max_input_tokens: int | None = None,
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE,
    provider_context_window: int | None = None,
    provider_max_input_tokens: int | None = None,
    context_tokenizer: Any = None,
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
            context_max_input_tokens=context_max_input_tokens,
            context_reserved_tool_tokens=context_reserved_tool_tokens,
            provider_context_window=provider_context_window,
            provider_max_input_tokens=provider_max_input_tokens,
            context_tokenizer=context_tokenizer,
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
    context_max_input_tokens: int | None = None,
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE,
    provider_context_window: int | None = None,
    provider_max_input_tokens: int | None = None,
    context_tokenizer: Any = None,
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
            context_max_input_tokens=context_max_input_tokens,
            context_reserved_tool_tokens=context_reserved_tool_tokens,
            provider_context_window=provider_context_window,
            provider_max_input_tokens=provider_max_input_tokens,
            context_tokenizer=context_tokenizer,
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
    context_max_input_tokens: int | None = None,
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE,
    provider_context_window: int | None = None,
    provider_max_input_tokens: int | None = None,
    context_tokenizer: Any = None,
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
            context_max_input_tokens=context_max_input_tokens,
            context_reserved_tool_tokens=context_reserved_tool_tokens,
            provider_context_window=provider_context_window,
            provider_max_input_tokens=provider_max_input_tokens,
            context_tokenizer=context_tokenizer,
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
    context_max_input_tokens: int | None = None,
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE,
    provider_context_window: int | None = None,
    provider_max_input_tokens: int | None = None,
    context_tokenizer: Any = None,
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
        context_max_input_tokens=context_max_input_tokens,
        context_reserved_tool_tokens=context_reserved_tool_tokens,
        provider_context_window=provider_context_window,
        provider_max_input_tokens=provider_max_input_tokens,
        context_tokenizer=context_tokenizer,
        provider_env=provider_env,
        model_env=model_env,
        default_model=default_model,
        default_max_new_tokens=default_max_new_tokens,
        default_timeout_seconds=default_timeout_seconds,
    )


def build_repo_task_proposal_route_runner(
    *,
    repo_root: Path | str,
    task_board_stem: str,
    task_board_dir: Path | str,
    artifact_namespace: str,
    task_header_prefix: str,
    prompt_intro: str,
    description: str,
    task_id_help: str,
    domain_outputs: Sequence[str] = (),
    test_output: str = DEFAULT_TASK_PROPOSAL_TEST_OUTPUT,
    requested_outputs: Sequence[str] | None = None,
    no_open_task_message: str = "No open task found.",
    task_board_option: str | None = None,
    hidden_task_board_options: Sequence[str] = (),
    hidden_standard_task_board_option: bool = False,
    include_dry_run_flag: bool = False,
    plan_stem: str | None = None,
    plan_dir: Path | str | None = None,
    artifact_dir: Path | str | None = None,
    artifact_root: Path | str = "data",
    artifact_leaf: Path | str = "llm_router",
    bootstrap: BootstrapCallback | None = None,
    runtime_package_names: Sequence[Path | str] = ("ipfs_accelerate", "ipfs_datasets"),
    runtime_external_dir: Path | str = "external",
    runtime_primary_package_names: Sequence[Path | str] | None = None,
    runtime_env_var: str = "PYTHONPATH",
    open_statuses: Sequence[str] = DEFAULT_OPEN_TASK_STATUSES,
    plan_char_limit: int = 40000,
    context_max_input_tokens: int | None = None,
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE,
    provider_context_window: int | None = None,
    provider_max_input_tokens: int | None = None,
    context_tokenizer: Any = None,
    provider_env: str = "IPFS_DATASETS_PY_LLM_PROVIDER",
    model_env: str = "IPFS_DATASETS_PY_LLM_MODEL",
    default_model: str = "gpt-5.3-codex-spark",
    default_max_new_tokens: int = 2048,
    default_timeout_seconds: int = 300,
) -> ConfiguredTaskProposalRouterRunner:
    """Build a repo task-proposal runner from standard route inputs."""

    from .wrapper_utils import task_board_path_option

    route_paths = build_task_proposal_route_paths(
        repo_root=repo_root,
        task_board_stem=task_board_stem,
        task_board_dir=task_board_dir,
        plan_stem=plan_stem,
        plan_dir=plan_dir,
        artifact_dir=artifact_dir,
        artifact_namespace=artifact_namespace,
        artifact_root=artifact_root,
        artifact_leaf=artifact_leaf,
    )
    effective_task_board_option = task_board_option or task_board_path_option()
    effective_hidden_task_board_options = tuple(hidden_task_board_options)
    standard_task_board_option = task_board_path_option()
    if (
        hidden_standard_task_board_option
        and standard_task_board_option != effective_task_board_option
        and standard_task_board_option not in effective_hidden_task_board_options
    ):
        effective_hidden_task_board_options = (
            *effective_hidden_task_board_options,
            standard_task_board_option,
        )
    effective_requested_outputs = tuple(
        requested_outputs
        if requested_outputs is not None
        else standard_task_proposal_requested_outputs(*domain_outputs, test_output=test_output)
    )
    return build_repo_task_proposal_router_runner(
        repo_root=repo_root,
        task_board_path=route_paths.task_board_path,
        task_header_prefix=task_header_prefix,
        plan_path=route_paths.plan_path,
        artifact_dir=route_paths.artifact_dir,
        prompt_intro=prompt_intro,
        requested_outputs=effective_requested_outputs,
        description=description,
        task_id_help=task_id_help,
        no_open_task_message=no_open_task_message,
        task_board_option=effective_task_board_option,
        hidden_task_board_options=effective_hidden_task_board_options,
        include_dry_run_flag=include_dry_run_flag,
        bootstrap=bootstrap,
        runtime_package_names=runtime_package_names,
        runtime_external_dir=runtime_external_dir,
        runtime_primary_package_names=runtime_primary_package_names,
        runtime_env_var=runtime_env_var,
        open_statuses=open_statuses,
        plan_char_limit=plan_char_limit,
        context_max_input_tokens=context_max_input_tokens,
        context_reserved_tool_tokens=context_reserved_tool_tokens,
        provider_context_window=provider_context_window,
        provider_max_input_tokens=provider_max_input_tokens,
        context_tokenizer=context_tokenizer,
        provider_env=provider_env,
        model_env=model_env,
        default_model=default_model,
        default_max_new_tokens=default_max_new_tokens,
        default_timeout_seconds=default_timeout_seconds,
    )


def build_repo_task_proposal_route_runner_from_spec(
    *,
    repo_root: Path | str,
    route_spec: TaskProposalRouteSpec,
    bootstrap: BootstrapCallback | None = None,
) -> ConfiguredTaskProposalRouterRunner:
    """Build a repo task-proposal runner from a reusable route spec."""

    return build_repo_task_proposal_route_runner(
        repo_root=repo_root,
        task_board_stem=route_spec.task_board_stem,
        task_board_dir=route_spec.task_board_dir,
        artifact_namespace=route_spec.artifact_namespace,
        task_header_prefix=route_spec.task_header_prefix,
        prompt_intro=route_spec.prompt_intro,
        description=route_spec.description,
        task_id_help=route_spec.task_id_help,
        domain_outputs=route_spec.domain_outputs,
        test_output=route_spec.test_output,
        requested_outputs=route_spec.requested_outputs,
        no_open_task_message=route_spec.no_open_task_message,
        task_board_option=route_spec.task_board_option,
        hidden_task_board_options=route_spec.hidden_task_board_options,
        hidden_standard_task_board_option=route_spec.hidden_standard_task_board_option,
        include_dry_run_flag=route_spec.include_dry_run_flag,
        plan_stem=route_spec.plan_stem,
        plan_dir=route_spec.plan_dir,
        artifact_dir=route_spec.artifact_dir,
        artifact_root=route_spec.artifact_root,
        artifact_leaf=route_spec.artifact_leaf,
        bootstrap=bootstrap if bootstrap is not None else route_spec.bootstrap,
        runtime_package_names=route_spec.runtime_package_names,
        runtime_external_dir=route_spec.runtime_external_dir,
        runtime_primary_package_names=route_spec.runtime_primary_package_names,
        runtime_env_var=route_spec.runtime_env_var,
        open_statuses=route_spec.open_statuses,
        plan_char_limit=route_spec.plan_char_limit,
        context_max_input_tokens=route_spec.context_max_input_tokens,
        context_reserved_tool_tokens=route_spec.context_reserved_tool_tokens,
        provider_context_window=route_spec.provider_context_window,
        provider_max_input_tokens=route_spec.provider_max_input_tokens,
        context_tokenizer=route_spec.context_tokenizer,
        provider_env=route_spec.provider_env,
        model_env=route_spec.model_env,
        default_model=route_spec.default_model,
        default_max_new_tokens=route_spec.default_max_new_tokens,
        default_timeout_seconds=route_spec.default_timeout_seconds,
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


def _router_tree_id(repo_root: Path, *, fallback_material: str) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD^{commit}"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    value = str(result.stdout or "").strip()
    if result.returncode == 0 and value:
        return value
    return "tree:sha256:" + hashlib.sha256(
        fallback_material.encode("utf-8")
    ).hexdigest()


def _router_repository_id(repo_root: Path) -> str:
    """Return a stable, non-secret repository identity for context receipts."""

    resolved = repo_root.resolve()
    git_dir = subprocess.run(
        ["git", "rev-parse", "--git-common-dir"],
        cwd=resolved,
        text=True,
        capture_output=True,
        check=False,
    )
    raw_git_dir = str(git_dir.stdout or "").strip()
    material = str(
        (
            Path(raw_git_dir)
            if Path(raw_git_dir).is_absolute()
            else resolved / raw_git_dir
        ).resolve()
    ) if git_dir.returncode == 0 and raw_git_dir else str(resolved)
    return "repository:sha256:" + hashlib.sha256(
        material.encode("utf-8")
    ).hexdigest()


def _context_artifact_path(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        # Absolute paths are never copied into provider references or receipts.
        return ""


def compile_task_proposal_context(
    config: TaskProposalRouterConfig,
    *,
    task: object,
    plan_text: str,
    repository_tree_id: str,
    context_id: str,
    max_new_tokens: int,
) -> ContextCompileResult:
    """Compile one immutable planning core plus ranked roadmap chunks."""

    if not isinstance(plan_text, str):
        raise TaskProposalRouterError("plan text must be a string")
    task_id = _task_value(task, "task_id")
    allowed_paths = tuple(
        sorted(
            {
                _normalize_proposal_path(path)
                for path in _task_values(task, "outputs")
            }
        )
    )
    validation_commands = tuple(_task_values(task, "validation"))
    # Existing wrappers own their domain-specific wording.  Calling them with
    # no roadmap material captures that policy without allowing the wrapper's
    # historical character slice to select evidence.
    planning_policy = config.prompt_builder(task, "").strip()
    policy_payload = {
        "instructions": planning_policy,
        "response_contract": TASK_IMPLEMENTATION_PROPOSAL_SCHEMA,
        "schema": TASK_IMPLEMENTATION_PROPOSAL_JSON_SCHEMA,
    }
    policy_revision = "sha256:" + hashlib.sha256(
        json.dumps(
            policy_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    repository_id = _router_repository_id(config.repo_root)
    roadmap_references = build_text_context_references(
        plan_text,
        reference_prefix="roadmap",
        kind="roadmap-chunk",
        path=_context_artifact_path(config.plan_path, config.repo_root),
        repository_id=repository_id,
        tree_id=repository_tree_id,
        priority=100,
        chunk_bytes=6_144,
        coverage_ids=(task_id,),
    )
    configured_input = config.context_max_input_tokens
    if configured_input is None:
        legacy_hint = max(0, int(config.plan_char_limit))
        configured_input = max(
            2_048,
            min(
                DEFAULT_PLANNING_CONTEXT_INPUT_TOKENS,
                (legacy_hint + 3) // 4 + 2_048,
            ),
        )
    budget = ContextBudget(
        max_input_tokens=max(1, int(configured_input)),
        reserved_output_tokens=max(0, int(max_new_tokens)),
        reserved_tool_tokens=max(
            0, int(config.context_reserved_tool_tokens)
        ),
        max_items=256,
        max_item_bytes=16_384,
        max_serialized_bytes=262_144,
        max_depth=12,
        max_text_bytes=8_192,
    )
    compiler = ContextCompiler(
        budget,
        tokenizer=config.context_tokenizer,
        provider_context_window=config.provider_context_window,
        provider_max_input_tokens=config.provider_max_input_tokens,
    )
    return compiler.compile(
        repository_id=repository_id,
        tree_id=repository_tree_id,
        objective_id=task_id,
        objective_revision=context_id,
        policy_id="policy:task-proposal-router",
        policy_revision=policy_revision,
        caller="agent-supervisor:task-proposal-router",
        stage="planning",
        goal={
            "task_id": task_id,
            "title": _task_value(task, "title"),
            "priority": _task_value(task, "priority"),
            "track": _task_value(task, "track"),
            "planning_policy": planning_policy,
        },
        authority={
            "repository_tree_id": repository_tree_id,
            "context_id": context_id,
            "allowed_paths": allowed_paths,
            "validation_commands": validation_commands,
            "validation_commands_only": True,
            "proof_authoritative": False,
            "completion_authoritative": False,
        },
        scope={
            "depends_on": tuple(_task_values(task, "depends_on")),
            "outputs": allowed_paths,
        },
        acceptance={
            "criteria": _task_value(task, "acceptance") or "none listed",
            "validation_commands": validation_commands,
            "response_rules": (
                "Return exactly one JSON object with no Markdown or extra fields.",
                "Files and validation commands must exactly match frozen authority.",
            ),
            "response_schema": TASK_IMPLEMENTATION_PROPOSAL_JSON_SCHEMA,
        },
        evidence=roadmap_references,
    )


def _proposal_json_depth(value: Any, depth: int = 0) -> int:
    if isinstance(value, Mapping):
        return max(
            (depth, *(_proposal_json_depth(item, depth + 1) for item in value.values()))
        )
    if isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        return max(
            (depth, *(_proposal_json_depth(item, depth + 1) for item in value))
        )
    return depth


def _normalize_proposal_path(value: Any) -> str:
    path = str(value or "").strip().replace("\\", "/")
    while path.startswith("./"):
        path = path[2:]
    if (
        not path
        or path.startswith("/")
        or "\0" in path
        or ".." in Path(path).parts
        or path == ".git"
        or path.startswith(".git/")
    ):
        raise TaskProposalRouterError(
            "proposal contains an unsafe path",
            reason_code="unsafe_path",
        )
    return path


def parse_task_implementation_proposal(
    text: str,
    *,
    expected_task_id: str,
    expected_repository_tree_id: str,
    expected_context_id: str,
    allowed_paths: Sequence[str],
    allowed_validation_commands: Sequence[str],
) -> dict[str, Any]:
    """Parse and fail-close one provider proposal before artifact persistence."""

    if not isinstance(text, str) or not text.strip():
        raise TaskProposalRouterError(
            "llm_router returned an empty proposal",
            reason_code="invalid_schema",
        )
    if len(text.encode("utf-8", errors="surrogatepass")) > TASK_PROPOSAL_MAX_RESPONSE_BYTES:
        raise TaskProposalRouterError(
            "llm_router proposal exceeds the output bound",
            reason_code="output_too_large",
        )

    def reject_duplicate_fields(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON field {key!r}")
            value[key] = item
        return value

    try:
        payload = json.loads(
            text,
            object_pairs_hook=reject_duplicate_fields,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as exc:
        raise TaskProposalRouterError(
            "llm_router proposal is not strict JSON",
            reason_code="invalid_schema",
        ) from exc
    if not isinstance(payload, Mapping):
        raise TaskProposalRouterError(
            "llm_router proposal must be a JSON object",
            reason_code="invalid_schema",
        )
    if _proposal_json_depth(payload) > TASK_PROPOSAL_MAX_JSON_DEPTH:
        raise TaskProposalRouterError(
            "llm_router proposal exceeds the nesting bound",
            reason_code="output_too_deep",
        )
    required = set(TASK_IMPLEMENTATION_PROPOSAL_JSON_SCHEMA["required"])
    fields = {str(key) for key in payload}
    if fields != required:
        reason = "missing_required_field" if required - fields else "invalid_schema"
        raise TaskProposalRouterError(
            "llm_router proposal fields do not match the versioned schema",
            reason_code=reason,
        )
    if (
        payload.get("schema") != TASK_IMPLEMENTATION_PROPOSAL_SCHEMA
        or str(payload.get("proposal_version") or "") != "1"
    ):
        raise TaskProposalRouterError(
            "unsupported task implementation proposal version",
            reason_code="invalid_schema",
        )
    for field_name, expected, reason_code in (
        ("task_id", expected_task_id, "authority_mismatch"),
        (
            "repository_tree_id",
            expected_repository_tree_id,
            "stale_baseline",
        ),
        ("context_id", expected_context_id, "context_mismatch"),
    ):
        if str(payload.get(field_name) or "") != str(expected):
            raise TaskProposalRouterError(
                f"proposal {field_name} does not match frozen authority",
                reason_code=reason_code,
            )

    normalized_allowed = tuple(
        sorted({_normalize_proposal_path(path) for path in allowed_paths})
    )
    raw_files = payload.get("files")
    if (
        not isinstance(raw_files, list)
        or not raw_files
        or len(raw_files) > TASK_PROPOSAL_MAX_FILES
    ):
        raise TaskProposalRouterError(
            "proposal files must be a bounded non-empty array",
            reason_code="missing_required_field",
        )
    normalized_files: list[dict[str, Any]] = []
    for item in raw_files:
        if not isinstance(item, Mapping) or set(item) != {
            "path",
            "operation",
            "rationale_references",
        }:
            raise TaskProposalRouterError(
                "proposal file operations do not match the schema",
                reason_code="invalid_schema",
            )
        path = _normalize_proposal_path(item.get("path"))
        operation = str(item.get("operation") or "").strip().lower()
        references = item.get("rationale_references")
        if operation not in TASK_PROPOSAL_OPERATIONS:
            raise TaskProposalRouterError(
                "proposal contains an unsupported file operation",
                reason_code="operation_mismatch",
            )
        if operation == "delete" and (
            path.startswith(("test/", "tests/"))
            or path.rsplit("/", 1)[-1].startswith("test_")
        ):
            raise TaskProposalRouterError(
                "proposal cannot delete tests",
                reason_code="test_deletion_forbidden",
            )
        if (
            not isinstance(references, list)
            or not references
            or any(not isinstance(value, str) or not value.strip() for value in references)
        ):
            raise TaskProposalRouterError(
                "every file operation requires rationale references",
                reason_code="missing_required_field",
            )
        normalized_files.append(
            {
                "path": path,
                "operation": operation,
                "rationale_references": sorted(
                    {str(value).strip() for value in references}
                ),
            }
        )
    proposed_paths = tuple(sorted(item["path"] for item in normalized_files))
    if len(set(proposed_paths)) != len(proposed_paths) or proposed_paths != normalized_allowed:
        raise TaskProposalRouterError(
            "proposal files do not exactly match task-owned outputs",
            reason_code="path_outside_scope",
        )

    validation_plan = payload.get("validation_plan")
    if (
        not isinstance(validation_plan, list)
        or any(not isinstance(value, str) or not value.strip() for value in validation_plan)
        or tuple(str(value).strip() for value in validation_plan)
        != tuple(str(value).strip() for value in allowed_validation_commands)
    ):
        raise TaskProposalRouterError(
            "proposal validation plan differs from task authority",
            reason_code="command_forbidden",
        )
    risks = payload.get("risks")
    if (
        not isinstance(risks, list)
        or not risks
        or any(not isinstance(value, str) or not value.strip() for value in risks)
    ):
        raise TaskProposalRouterError(
            "proposal requires a bounded risk assessment",
            reason_code="missing_required_field",
        )
    claims = payload.get("authority_claims")
    expected_claims = {
        "allowed_paths": list(normalized_allowed),
        "validation_commands_only": True,
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    if not isinstance(claims, Mapping) or dict(claims) != expected_claims:
        raise TaskProposalRouterError(
            "proposal contains detached or forged authority claims",
            reason_code="forged_authority_claim",
        )
    return {
        "schema": TASK_IMPLEMENTATION_PROPOSAL_SCHEMA,
        "proposal_version": "1",
        "task_id": expected_task_id,
        "repository_tree_id": expected_repository_tree_id,
        "context_id": expected_context_id,
        "files": sorted(normalized_files, key=lambda item: item["path"]),
        "validation_plan": list(allowed_validation_commands),
        "risks": [str(value).strip() for value in risks],
        "authority_claims": expected_claims,
    }


def build_task_implementation_proposal_contract(
    *,
    task: object,
    repository_tree_id: str,
    context_id: str,
) -> str:
    allowed_paths = tuple(
        sorted({_normalize_proposal_path(path) for path in _task_values(task, "outputs")})
    )
    authority = {
        "task_id": _task_value(task, "task_id"),
        "repository_tree_id": repository_tree_id,
        "context_id": context_id,
        "allowed_paths": allowed_paths,
        "validation_commands": _task_values(task, "validation"),
        "proof_authoritative": False,
        "completion_authoritative": False,
    }
    return "\n".join(
        (
            "",
            "Strict proposal envelope:",
            "Return exactly one JSON object. Do not return Markdown, prose, comments, shell wrappers, or extra fields.",
            "Files must exactly equal the frozen allowed_paths. Every file needs an operation and rationale references.",
            "validation_plan must exactly equal the frozen validation commands; do not invent or combine commands.",
            "Authority claims must repeat allowed_paths, set validation_commands_only=true, and set proof_authoritative=false and completion_authoritative=false.",
            "Frozen authority:",
            json.dumps(authority, indent=2, sort_keys=True),
            "Required JSON schema:",
            json.dumps(
                TASK_IMPLEMENTATION_PROPOSAL_JSON_SCHEMA,
                indent=2,
                sort_keys=True,
            ),
        )
    )


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
    plan_text = config.plan_path.read_text(encoding="utf-8")
    context_id = str(
        getattr(selected, "canonical_task_cid", "")
        or getattr(selected, "canonical_task_key", "")
        or _task_value(selected, "task_id")
    ).strip()
    repository_tree_id = _router_tree_id(
        config.repo_root,
        fallback_material="\0".join(
            (
                _task_value(selected, "task_id"),
                context_id,
                plan_text,
            )
        ),
    )
    compiled_context = compile_task_proposal_context(
        config,
        task=selected,
        plan_text=plan_text,
        repository_tree_id=repository_tree_id,
        context_id=context_id,
        max_new_tokens=max_new_tokens,
    )
    prompt = render_context_capsule(compiled_context.capsule)
    payload: dict[str, object] = {
        "task_id": _task_value(selected, "task_id"),
        "title": _task_value(selected, "title"),
        "provider": provider or None,
        "model": model,
        "prompt_chars": len(prompt),
        "generate": bool(generate),
        "llm_router_importable": True,
        "context_capsule_id": compiled_context.capsule.capsule_id,
        "context_input_tokens": compiled_context.capsule.input_tokens,
        "context_input_limit": compiled_context.receipt.effective_input_limit,
        "context_truncated": compiled_context.capsule.truncated,
        "context_estimator": compiled_context.receipt.estimator_name,
        "context_estimator_error_bps": (
            compiled_context.receipt.estimator_error_bps
        ),
        "context_decisions": [
            decision.to_dict() for decision in compiled_context.decisions
        ],
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
    raw_proposal = call_llm_router(prompt, invocation)
    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    task_name = (_task_value(selected, "task_id") or "task").lower()
    context_receipt_path = (
        config.artifact_dir / f"{task_name}-context-receipt.json"
    )
    context_receipt_path.write_text(
        compiled_context.receipt.to_json() + "\n",
        encoding="utf-8",
    )
    payload["context_receipt"] = _artifact_relative_path(
        context_receipt_path, config.repo_root
    )
    try:
        proposal = parse_task_implementation_proposal(
            raw_proposal,
            expected_task_id=_task_value(selected, "task_id"),
            expected_repository_tree_id=repository_tree_id,
            expected_context_id=context_id,
            allowed_paths=_task_values(selected, "outputs"),
            allowed_validation_commands=_task_values(selected, "validation"),
        )
    except TaskProposalRouterError as exc:
        rejection_path = (
            config.artifact_dir / f"{task_name}-proposal-rejection.json"
        )
        rejection_path.write_text(
            json.dumps(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/"
                        "task-proposal-rejection@1"
                    ),
                    "accepted": False,
                    "task_id": _task_value(selected, "task_id"),
                    "repository_tree_id": repository_tree_id,
                    "context_id": context_id,
                    "reason_codes": [exc.reason_code],
                    "proof_authoritative": False,
                    "completion_authoritative": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise TaskProposalRouterError(
            f"{exc}; compact rejection: "
            f"{_artifact_relative_path(rejection_path, config.repo_root)}",
            reason_code=exc.reason_code,
        ) from exc
    output_path = config.artifact_dir / f"{task_name}-proposal.json"
    output_path.write_text(
        json.dumps(proposal, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    payload["artifact"] = _artifact_relative_path(output_path, config.repo_root)
    payload["proposal_schema"] = TASK_IMPLEMENTATION_PROPOSAL_SCHEMA
    payload["repository_tree_id"] = repository_tree_id
    payload["context_id"] = context_id
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


# Structured objective-plan routing -------------------------------------------------

StructuredRouter = Callable[[str], str]
FallbackPlanner = Callable[[object, int], Sequence[PlanBranch | Mapping[str, Any]]]


@dataclass(frozen=True)
class StructuredPlanRouterConfig:
    """Runtime settings for one structured ``llm_router`` planning call."""

    repo_root: Path = field(default_factory=Path.cwd)
    provider: str | None = None
    model: str = "gpt-5.3-codex-spark"
    branch_count: int = 3
    max_new_tokens: int = 4096
    timeout_seconds: int = 300
    allow_local_fallback: bool = False
    temperature: float = 0.1
    context_max_input_tokens: int = DEFAULT_PLANNING_CONTEXT_INPUT_TOKENS
    context_reserved_tool_tokens: int = DEFAULT_PLANNING_CONTEXT_TOOL_RESERVE
    provider_context_window: int | None = None
    provider_max_input_tokens: int | None = None
    context_tokenizer: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.branch_count) < 1:
            raise ValueError("branch_count must be at least 1")
        if int(self.max_new_tokens) < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if int(self.timeout_seconds) < 1:
            raise ValueError("timeout_seconds must be at least 1")
        if int(self.context_max_input_tokens) < 1:
            raise ValueError("context_max_input_tokens must be at least 1")
        if int(self.context_reserved_tool_tokens) < 0:
            raise ValueError("context_reserved_tool_tokens must be non-negative")
        if not 0.0 <= float(self.temperature) <= 2.0:
            raise ValueError("temperature must be in [0, 2]")


@dataclass(frozen=True)
class PlanRoutingResult:
    """Schema-validated branches plus auditable router/fallback provenance."""

    branches: tuple[PlanBranch, ...]
    used_fallback: bool
    router_error: str | None = None
    raw_response: str | None = None

    def __post_init__(self) -> None:
        if not self.branches:
            raise ValueError("a routing result must contain at least one plan branch")

    @property
    def router_succeeded(self) -> bool:
        return not self.used_fallback

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        encode = PlanBranch.to_profile_g_dict if profile_g else PlanBranch.to_dict
        response_bytes = (
            self.raw_response.encode("utf-8", errors="surrogatepass")
            if self.raw_response is not None
            else b""
        )
        return {
            "branches": [encode(branch) for branch in self.branches],
            "used_fallback": self.used_fallback,
            "router_succeeded": self.router_succeeded,
            "router_error": self.router_error,
            "response_bytes": len(response_bytes),
            "response_sha256": (
                "sha256:" + hashlib.sha256(response_bytes).hexdigest()
                if self.raw_response is not None
                else ""
            ),
        }

    def to_profile_g_dict(self) -> dict[str, Any]:
        return self.to_dict(profile_g=True)


@dataclass(frozen=True)
class AnalysisProposalRoutingResult:
    """Bounded semantic routing result with explicit fail-closed provenance."""

    proposals: tuple[AnalysisProposal, ...]
    evaluation: AnalysisProposalEvaluation
    router_evaluation: AnalysisProposalEvaluation | None
    used_fallback: bool
    analysis_inconclusive: bool
    router_calls: int = 0
    router_retries: int = 0
    reserved_tokens: int = 0
    router_error: str | None = None
    raw_responses: tuple[str, ...] = ()
    router_call_timestamps: tuple[float, ...] = ()
    limit_reason: str = ""

    @property
    def router_succeeded(self) -> bool:
        return not self.used_fallback and not self.analysis_inconclusive

    @property
    def accepted(self) -> tuple[AnalysisProposal, ...]:
        return self.evaluation.accepted

    @property
    def rejected(self) -> tuple[Any, ...]:
        return self.evaluation.rejected

    def to_dict(self, *, profile_g: bool = False) -> dict[str, Any]:
        response_bytes = tuple(
            item.encode("utf-8", errors="surrogatepass")
            for item in self.raw_responses
        )
        return {
            "proposals": [item.to_dict(profile_g=profile_g) for item in self.proposals],
            "evaluation": self.evaluation.to_dict(profile_g=profile_g),
            "router_evaluation": (
                self.router_evaluation.to_dict(profile_g=profile_g)
                if self.router_evaluation is not None
                else None
            ),
            "used_fallback": self.used_fallback,
            "analysis_inconclusive": self.analysis_inconclusive,
            "router_succeeded": self.router_succeeded,
            "router_calls": self.router_calls,
            "router_retries": self.router_retries,
            "reserved_tokens": self.reserved_tokens,
            "router_error": self.router_error,
            "response_count": len(response_bytes),
            "response_bytes": sum(len(item) for item in response_bytes),
            "response_sha256": [
                "sha256:" + hashlib.sha256(item).hexdigest()
                for item in response_bytes
            ],
            "router_call_timestamps": list(self.router_call_timestamps),
            "limit_reason": self.limit_reason,
        }


def _object_value(value: object, *names: str) -> Any:
    for name in names:
        if isinstance(value, Mapping) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _values(value: object, *names: str) -> tuple[str, ...]:
    raw = _object_value(value, *names)
    if raw is None:
        return ()
    if isinstance(raw, str):
        return tuple(item.strip() for item in raw.split(",") if item.strip())
    if isinstance(raw, Sequence):
        return tuple(str(item).strip() for item in raw if str(item).strip())
    return (str(raw).strip(),) if str(raw).strip() else ()


def _jsonable_subgoal(subgoal: object) -> dict[str, Any]:
    if isinstance(subgoal, Mapping):
        source = dict(subgoal)
    elif hasattr(subgoal, "to_dict") and callable(getattr(subgoal, "to_dict")):
        converted = subgoal.to_dict()
        source = (
            dict(converted)
            if isinstance(converted, Mapping)
            else {"description": str(converted)}
        )
    elif hasattr(subgoal, "__dict__"):
        source = dict(vars(subgoal))
    else:
        source = {"description": str(subgoal)}
    names = (
        "subgoal_cid",
        "goal_id",
        "task_id",
        "title",
        "summary",
        "goal",
        "missing_evidence",
        "acceptance",
        "outputs",
        "predicted_files",
        "ast_symbols",
        "predicted_symbols",
        "dependencies",
        "depends_on",
        "validation",
        "validation_commands",
        "interfaces",
        "submodules",
    )
    return {name: source[name] for name in names if name in source}


def _planning_capsule_prompt(
    *,
    config: StructuredPlanRouterConfig,
    objective_id: str,
    goal: Mapping[str, Any],
    authority: Mapping[str, Any],
    scope: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    evidence: Sequence[Any] = (),
    evidence_text: str = "",
    evidence_kind: str = "planning-evidence",
    evidence_coverage_ids: Sequence[str] = (),
    policy_id: str,
) -> str:
    """Compile the canonical provider input shared by planning routes."""

    repository_id = _router_repository_id(config.repo_root)
    invariant = {
        "goal": goal,
        "authority": authority,
        "scope": scope,
        "acceptance": acceptance,
    }
    revision = "sha256:" + hashlib.sha256(
        json.dumps(
            invariant,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    tree_id = _router_tree_id(
        config.repo_root,
        fallback_material=revision,
    )
    policy_revision = "sha256:" + hashlib.sha256(
        json.dumps(
            {
                "authority": authority,
                "acceptance": acceptance,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()
    ranked_evidence = tuple(evidence)
    if evidence_text:
        ranked_evidence = (
            *ranked_evidence,
            *build_text_context_references(
                evidence_text,
                reference_prefix="planning-evidence",
                kind=evidence_kind,
                repository_id=repository_id,
                tree_id=tree_id,
                priority=100,
                chunk_bytes=6_144,
                coverage_ids=evidence_coverage_ids,
            ),
        )
    compiler = ContextCompiler(
        ContextBudget(
            max_input_tokens=int(config.context_max_input_tokens),
            reserved_output_tokens=int(config.max_new_tokens),
            reserved_tool_tokens=int(config.context_reserved_tool_tokens),
            max_items=256,
            max_item_bytes=16_384,
            max_serialized_bytes=262_144,
            max_depth=12,
            max_text_bytes=8_192,
        ),
        tokenizer=config.context_tokenizer,
        provider_context_window=config.provider_context_window,
        provider_max_input_tokens=config.provider_max_input_tokens,
    )
    result = compiler.compile(
        repository_id=repository_id,
        tree_id=tree_id,
        objective_id=objective_id,
        objective_revision=revision,
        policy_id=policy_id,
        policy_revision=policy_revision,
        caller="agent-supervisor:structured-planning-router",
        stage="planning",
        goal=goal,
        authority=authority,
        scope=scope,
        acceptance=acceptance,
        evidence=ranked_evidence,
    )
    return render_context_capsule(result.capsule)


def build_structured_plan_prompt(
    subgoal: object,
    branch_count: int = 3,
    *,
    config: StructuredPlanRouterConfig | None = None,
) -> str:
    """Build a token-budgeted strict request for objective branch generation."""

    count = int(branch_count)
    if count < 1:
        raise ValueError("branch_count must be at least 1")
    context = _jsonable_subgoal(subgoal)
    objective_id = str(
        context.get("task_id")
        or context.get("goal_id")
        or context.get("subgoal_cid")
        or "structured-plan"
    )
    instructions = (
        "Generate alternative implementation plan branches for this scheduler subgoal.",
        f"Return exactly {count} materially distinct branches.",
        "Return JSON only: no Markdown fence, prose, comments, NaN, or Infinity.",
        "All files must be repository-relative and source must be 'llm_router'.",
        "estimated_cost is non-negative; risk and expected_objective_delta are in [0, 1].",
        "validation_proof states observable success evidence expected from the commands.",
    )
    return _planning_capsule_prompt(
        config=config or StructuredPlanRouterConfig(branch_count=count),
        objective_id=objective_id,
        policy_id="policy:structured-plan-branch-router",
        goal={
            "subgoal": context,
            "requested_branch_count": count,
        },
        authority={
            "instructions": instructions,
            "provider_source": "llm_router",
            "repository_relative_files_only": True,
            "completion_authoritative": False,
        },
        scope={
            "predicted_files": context.get(
                "predicted_files", context.get("outputs", ())
            ),
            "predicted_symbols": context.get(
                "predicted_symbols", context.get("ast_symbols", ())
            ),
            "dependencies": context.get(
                "dependencies", context.get("depends_on", ())
            ),
        },
        acceptance={
            "response_schema": PLAN_BRANCH_JSON_SCHEMA,
            "exact_branch_count": count,
            "strict_json_only": True,
        },
    )


def _decode_router_json(text: str) -> Any:
    if not isinstance(text, str) or not text.strip():
        raise PlanBranchValidationError("llm_router returned an empty response")
    stripped = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced is not None:
        stripped = fenced.group(1).strip()
    else:
        embedded = re.findall(
            r"```(?:json)?\s*(.*?)\s*```",
            stripped,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if len(embedded) == 1:
            stripped = embedded[0].strip()
    try:
        return json.loads(
            stripped,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON number {value}")
            ),
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise PlanBranchValidationError(
            f"llm_router response is not valid JSON: {exc}"
        ) from exc


def parse_structured_plan_branches(text: str) -> tuple[PlanBranch, ...]:
    """Parse one complete router response, rejecting partial/mixed validity."""

    payload = _decode_router_json(text)
    if isinstance(payload, Mapping):
        unknown = sorted(str(key) for key in payload if key != "branches")
        if unknown:
            raise PlanBranchValidationError(
                f"unknown top-level plan fields: {', '.join(unknown)}"
            )
        if "branches" not in payload:
            raise PlanBranchValidationError(
                "router JSON object must contain 'branches'"
            )
        raw_branches = payload["branches"]
    else:
        raw_branches = payload
    if isinstance(raw_branches, (str, bytes)) or not isinstance(
        raw_branches, Sequence
    ):
        raise PlanBranchValidationError("branches must be a JSON array")
    if not raw_branches:
        raise PlanBranchValidationError(
            "branches must contain at least one candidate"
        )
    required_fields = set(
        PLAN_BRANCH_JSON_SCHEMA["properties"]["branches"]["items"]["required"]
    )
    branches_list: list[PlanBranch] = []
    for index, item in enumerate(raw_branches):
        if not isinstance(item, Mapping):
            raise PlanBranchValidationError(
                f"branches[{index}] must be a JSON object"
            )
        fields = {str(key) for key in item}
        missing = sorted(required_fields - fields)
        unknown = sorted(fields - required_fields)
        if missing:
            raise PlanBranchValidationError(
                f"branches[{index}] is missing required fields: {', '.join(missing)}"
            )
        if unknown:
            raise PlanBranchValidationError(
                f"branches[{index}] contains unknown fields: {', '.join(unknown)}"
            )
        branch = PlanBranch.from_dict(item)
        if branch.source != "llm_router":
            raise PlanBranchValidationError(
                f"branches[{index}].source must be 'llm_router'"
            )
        branches_list.append(branch)
    branches = tuple(branches_list)
    branch_ids = [branch.branch_id for branch in branches]
    duplicates = sorted(
        {item for item in branch_ids if branch_ids.count(item) > 1}
    )
    if duplicates:
        raise PlanBranchValidationError(
            f"duplicate branch ids: {', '.join(duplicates)}"
        )
    return branches


def _compact_ast_planning_evidence(
    ast_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Allowlist bounded AST coverage facts without source or graph bodies."""

    if not isinstance(ast_evidence, Mapping):
        return {}
    scalar_fields = (
        "analyzer_version",
        "healthy",
        "complete",
        "expected_file_count",
        "scanned_file_count",
        "parsed_file_count",
        "reused_file_count",
        "parse_failure_count",
        "source_bytes",
        "ast_truncated_count",
        "confidence",
        "novelty",
        "limit_reason",
        "elapsed_seconds",
    )
    sequence_fields = ("covered_terms", "uncovered_terms", "objective_terms")
    compact: dict[str, Any] = {
        name: ast_evidence[name]
        for name in scalar_fields
        if isinstance(ast_evidence.get(name), (str, int, float, bool))
    }
    for name in sequence_fields:
        raw = ast_evidence.get(name)
        if isinstance(raw, Sequence) and not isinstance(
            raw, (str, bytes, bytearray)
        ):
            compact[name] = [
                str(item) for item in raw if isinstance(item, (str, int, float))
            ]
    pipeline = ast_evidence.get("analysis_pipeline")
    if isinstance(pipeline, Mapping):
        pipeline_fields = (
            "status",
            "reason_code",
            "error_type",
            "result_id",
            "cache_status",
            "cache_lookup_status",
            "ast_index_id",
            "retrieval_response_id",
            "safe_for_completion_reasoning",
            "nomination_only",
        )
        compact["analysis_pipeline"] = {
            name: pipeline[name]
            for name in pipeline_fields
            if isinstance(pipeline.get(name), (str, int, float, bool))
        }
    return compact


def build_analysis_proposal_prompt(
    context: object,
    *,
    objective_terms: Sequence[str],
    ast_evidence: Mapping[str, Any] | None = None,
    proposal_count: int = 3,
    config: StructuredPlanRouterConfig | None = None,
) -> str:
    """Build a token-budgeted, schema-constrained analysis request."""

    count = int(proposal_count)
    if count < 1:
        raise ValueError("proposal_count must be at least 1")
    terms = tuple(dict.fromkeys(str(item).strip() for item in objective_terms if str(item).strip()))
    if not terms:
        raise ValueError("objective_terms must contain at least one term")
    planning_context = _jsonable_subgoal(context)
    objective_id = str(
        planning_context.get("task_id")
        or planning_context.get("goal_id")
        or planning_context.get("subgoal_cid")
        or "analysis-proposal"
    )
    instructions = (
        "Propose bounded tasks for objective terms still uncovered after static and AST analysis.",
        f"Return between 1 and {count} materially distinct proposals.",
        "Return JSON only: no Markdown fence, prose, comments, NaN, or Infinity.",
        "Each nested branch source must be 'llm_router'; confidence and novelty are in [0, 1].",
        "Use only supplied objective terms and repository-relative predicted files.",
        "Do not claim that the objective or repository is exhausted.",
    )
    compact_ast = _compact_ast_planning_evidence(ast_evidence)
    evidence_text = (
        json.dumps(
            compact_ast,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        if compact_ast
        else ""
    )
    return _planning_capsule_prompt(
        config=config or StructuredPlanRouterConfig(branch_count=count),
        objective_id=objective_id,
        policy_id="policy:analysis-proposal-router",
        goal={
            "planning_context": planning_context,
            "objective_terms": terms,
            "maximum_proposal_count": count,
        },
        authority={
            "instructions": instructions,
            "provider_source": "llm_router",
            "repository_relative_files_only": True,
            "completion_authoritative": False,
            "exhaustion_authoritative": False,
        },
        scope={
            "objective_terms": terms,
            "predicted_files": planning_context.get(
                "predicted_files", planning_context.get("outputs", ())
            ),
            "predicted_symbols": planning_context.get(
                "predicted_symbols",
                planning_context.get("ast_symbols", ()),
            ),
        },
        acceptance={
            "response_schema": ANALYSIS_PROPOSAL_JSON_SCHEMA,
            "proposal_count": {"minimum": 1, "maximum": count},
            "strict_json_only": True,
        },
        evidence_text=evidence_text,
        evidence_kind="ast-coverage-summary",
        evidence_coverage_ids=terms,
    )


def parse_analysis_proposals(text: str) -> tuple[AnalysisProposal, ...]:
    """Parse a strict semantic proposal response as an all-or-nothing unit."""

    payload = _decode_router_json(text)
    if not isinstance(payload, Mapping):
        raise PlanBranchValidationError("analysis proposal response must be a JSON object")
    unknown = sorted(str(key) for key in payload if key != "proposals")
    if unknown:
        raise PlanBranchValidationError(
            f"unknown top-level analysis proposal fields: {', '.join(unknown)}"
        )
    raw = payload.get("proposals")
    if isinstance(raw, (str, bytes)) or not isinstance(raw, Sequence) or not raw:
        raise PlanBranchValidationError("proposals must be a non-empty JSON array")
    required = {"branch", "confidence", "novelty", "objective_terms"}
    for index, item in enumerate(raw):
        if not isinstance(item, Mapping):
            raise PlanBranchValidationError(f"proposals[{index}] must be a JSON object")
        fields = {str(key) for key in item}
        missing = sorted(required - fields)
        unknown_fields = sorted(fields - required)
        if missing:
            raise PlanBranchValidationError(
                f"proposals[{index}] is missing required fields: {', '.join(missing)}"
            )
        if unknown_fields:
            raise PlanBranchValidationError(
                f"proposals[{index}] contains unknown fields: {', '.join(unknown_fields)}"
            )
    proposals = tuple(AnalysisProposal.from_dict(item) for item in raw)
    branch_ids = [item.branch.branch_id for item in proposals]
    duplicates = sorted({item for item in branch_ids if branch_ids.count(item) > 1})
    if duplicates:
        raise PlanBranchValidationError(
            f"duplicate analysis proposal branch ids: {', '.join(duplicates)}"
        )
    for index, proposal in enumerate(proposals):
        if proposal.branch.source != "llm_router":
            raise PlanBranchValidationError(
                f"proposals[{index}].branch.source must be 'llm_router'"
            )
    return proposals


def deterministic_plan_branches(
    subgoal: object,
    branch_count: int = 1,
) -> tuple[PlanBranch, ...]:
    """Derive safe, deterministic branches without an LLM provider."""

    requested = max(1, int(branch_count))
    identifier = str(
        _object_value(subgoal, "task_id", "goal_id", "subgoal_cid", "id")
        or "subgoal"
    ).strip()
    safe_identifier = (
        re.sub(r"[^A-Za-z0-9._:-]+", "-", identifier).strip("-.") or "subgoal"
    )
    title = str(
        _object_value(subgoal, "title", "summary", "goal", "description")
        or identifier
    ).strip()
    files = _values(subgoal, "predicted_files", "outputs", "files") or (
        "objective-plan.unspecified",
    )
    symbols = _values(
        subgoal, "predicted_symbols", "ast_symbols", "symbols", "interfaces"
    ) or (re.sub(r"\W+", "_", safe_identifier).strip("_") or "objective_subgoal",)
    dependencies = _values(
        subgoal, "dependencies", "depends_on", "dependency_task_cids"
    )
    validations = _values(subgoal, "validation_commands", "validation") or (
        "git diff --check",
    )
    proof = tuple(f"{command} exits with status 0" for command in validations)
    variants = (
        ("focused", 1.0, 0.20, 0.70),
        ("incremental", 1.2, 0.15, 0.65),
        ("proof-first", 1.4, 0.10, 0.60),
    )
    results: list[PlanBranch] = []
    for index in range(requested):
        label, cost, risk, delta = variants[index % len(variants)]
        cycle = index // len(variants)
        results.append(
            PlanBranch(
                branch_id=f"fallback:{safe_identifier}:{label}-{index + 1}",
                summary=f"Deterministic {label} plan for {title}",
                predicted_files=files,
                predicted_symbols=symbols,
                dependencies=dependencies,
                validation_commands=validations,
                validation_proof=proof,
                estimated_cost=cost + cycle,
                risk=risk,
                expected_objective_delta=delta,
                source="deterministic_fallback",
            )
        )
    return tuple(results)


def _default_structured_router(
    prompt: str,
    config: StructuredPlanRouterConfig,
) -> str:
    from .todo_daemon.llm import LlmRouterInvocation, call_llm_router

    invocation = LlmRouterInvocation(
        repo_root=config.repo_root,
        model_name=config.model,
        provider=config.provider,
        allow_local_fallback=config.allow_local_fallback,
        timeout_seconds=config.timeout_seconds,
        max_new_tokens=config.max_new_tokens,
        temperature=config.temperature,
        reject_effective_provider_name=(
            None if config.allow_local_fallback else "local_hf"
        ),
    )
    return call_llm_router(prompt, invocation)


def generate_structured_plan_branches(
    subgoal: object,
    *,
    router: StructuredRouter | None = None,
    fallback_planner: FallbackPlanner | None = None,
    config: StructuredPlanRouterConfig | None = None,
    branch_count: int | None = None,
) -> PlanRoutingResult:
    """Generate validated branches, falling back without blocking ready work."""

    resolved_config = config or StructuredPlanRouterConfig()
    count = int(
        branch_count if branch_count is not None else resolved_config.branch_count
    )
    if count < 1:
        raise ValueError("branch_count must be at least 1")
    prompt = build_structured_plan_prompt(
        subgoal,
        count,
        config=resolved_config,
    )
    raw_response: str | None = None
    try:
        raw_response = (
            router(prompt)
            if router is not None
            else _default_structured_router(prompt, resolved_config)
        )
        branches = parse_structured_plan_branches(raw_response)
        if len(branches) != count:
            raise PlanBranchValidationError(
                f"llm_router returned {len(branches)} branches; expected exactly {count}"
            )
        return PlanRoutingResult(
            branches=branches,
            used_fallback=False,
            raw_response=raw_response,
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:1000]
        planner = fallback_planner or deterministic_plan_branches
        fallback_values = planner(subgoal, count)
        fallback_branches = tuple(
            item if isinstance(item, PlanBranch) else PlanBranch.from_dict(item)
            for item in fallback_values
        )
        if not fallback_branches:
            raise TaskProposalRouterError(
                "llm_router failed and fallback planner returned no branches: "
                f"{error}"
            ) from exc
        return PlanRoutingResult(
            branches=fallback_branches,
            used_fallback=True,
            router_error=error,
            raw_response=raw_response,
        )


def _deterministic_analysis_proposals(
    context: object,
    objective_terms: Sequence[str],
    count: int,
) -> tuple[AnalysisProposal, ...]:
    terms = tuple(dict.fromkeys(str(item).strip() for item in objective_terms if str(item).strip()))
    branches = deterministic_plan_branches(context, max(1, int(count)))
    return tuple(
        AnalysisProposal(
            branch=branch,
            # These values describe confidence in the fallback task shape,
            # not confidence that semantic analysis proved exhaustion.
            confidence=1.0,
            novelty=1.0,
            objective_terms=terms or ("unresolved objective",),
        )
        for branch in branches
    )


def analysis_proposals_to_objective_work(
    proposals: Iterable[AnalysisProposal | Mapping[str, Any]],
    *,
    parent_goal_id: str,
    depth: int = 1,
    kind: str = "task",
    estimated_tokens: int = 0,
    retry_count: int = 0,
) -> tuple[Any, ...]:
    """Project routed analysis candidates into canonical scheduler work.

    ``AnalysisProposal`` is deliberately a provider-boundary shape whereas
    ``ObjectiveWorkProposal`` is the durable objective-graph shape.  Keeping
    this conversion here prevents the daemon from depending on LLM-selected
    branch identifiers and ensures canonical identity is calculated from the
    actual evidence surface, files, symbols, and validation proof.
    """

    from .objective_graph import ObjectiveWorkProposal

    parent = str(parent_goal_id or "").strip()
    work: list[ObjectiveWorkProposal] = []
    for value in proposals:
        proposal = (
            value
            if isinstance(value, AnalysisProposal)
            else AnalysisProposal.from_dict(value)
        )
        branch = proposal.branch
        work.append(
            ObjectiveWorkProposal(
                kind=kind,
                title=branch.summary,
                parent_goal_id=parent,
                parent_objective_terms=proposal.objective_terms,
                # Validation proof describes the observable evidence gained
                # when the proposed commands succeed, so it is a stronger
                # delta than the branch's scalar utility estimate.
                expected_evidence_delta=branch.validation_proof,
                dependencies=branch.dependencies,
                predicted_files=branch.predicted_files,
                predicted_symbols=branch.predicted_symbols,
                validation_commands=branch.validation_commands,
                confidence=proposal.confidence,
                estimated_cost=branch.estimated_cost,
                novelty=proposal.novelty,
                depth=depth,
                estimated_tokens=estimated_tokens,
                retry_count=retry_count,
                source=branch.source,
                source_id=proposal.proposal_id,
                rationale=(
                    f"Expected objective delta {branch.expected_objective_delta:.6f}; "
                    f"risk {branch.risk:.6f}."
                ),
            )
        )
    return tuple(work)


def generate_analysis_proposals(
    context: object,
    *,
    objective_terms: Sequence[str],
    ast_evidence: Mapping[str, Any] | None = None,
    router: StructuredRouter | None = None,
    config: StructuredPlanRouterConfig | None = None,
    policy: Any = None,
    known_proposal_ids: Iterable[str] = (),
    router_calls_in_window: int | Iterable[Any] = 0,
    now: float | None = None,
    fallback_planner: Callable[[object, int], Sequence[Any]] | None = None,
) -> AnalysisProposalRoutingResult:
    """Route semantic proposals under rate, token, retry, and novelty caps.

    Any provider/schema/quality failure returns deterministic work while
    retaining ``analysis_inconclusive=True``. A fallback is useful scheduler
    input, but is never semantic evidence that the repository is exhausted.
    """

    from .analyzer_health import AnalysisEscalationPolicy

    limits = AnalysisEscalationPolicy.from_value(policy)
    resolved = config or StructuredPlanRouterConfig()
    desired = min(resolved.branch_count, max(1, limits.max_novel_proposals))
    prompt = build_analysis_proposal_prompt(
        context,
        objective_terms=objective_terms,
        ast_evidence=ast_evidence,
        proposal_count=desired,
        config=resolved,
    )
    now_epoch = float(time.time() if now is None else now)
    historical_timestamps: list[float] = []
    if isinstance(router_calls_in_window, int):
        historical_count = max(0, router_calls_in_window)
    else:
        cutoff = now_epoch - limits.router_window_seconds
        for item in router_calls_in_window:
            try:
                stamp = float(item.timestamp() if hasattr(item, "timestamp") else item)
            except (TypeError, ValueError, OverflowError, OSError):
                continue
            if cutoff <= stamp <= now_epoch:
                historical_timestamps.append(stamp)
        historical_count = len(historical_timestamps)
    calls_remaining = min(
        limits.max_router_calls,
        max(0, limits.router_calls_per_window - historical_count),
    )
    token_cost = min(int(resolved.max_new_tokens), limits.max_router_tokens)
    token_limited_calls = limits.max_router_tokens // max(1, token_cost)
    attempt_limit = min(1 + limits.max_router_retries, calls_remaining, token_limited_calls)
    errors: list[str] = []
    raw_responses: list[str] = []
    calls = 0
    last_evaluation = AnalysisProposalEvaluation((), (), None)
    proposals: tuple[AnalysisProposal, ...] = ()
    limit_reason = ""
    if attempt_limit <= 0:
        if calls_remaining <= 0:
            limit_reason = "router_rate_or_call_limit_reached"
        else:
            limit_reason = "router_token_limit_reached"
    for _attempt in range(attempt_limit):
        calls += 1
        try:
            raw = (
                router(prompt)
                if router is not None
                else _default_structured_router(
                    prompt,
                    replace(resolved, max_new_tokens=token_cost),
                )
            )
            raw_responses.append(str(raw))
            proposals = parse_analysis_proposals(raw)
            if len(proposals) > desired:
                raise PlanBranchValidationError(
                    f"llm_router returned {len(proposals)} proposals; maximum is {desired}"
                )
            last_evaluation = evaluate_analysis_proposals(
                proposals,
                objective_terms=objective_terms,
                known_proposal_ids=known_proposal_ids,
                min_confidence=limits.min_confidence,
                min_novelty=limits.min_novelty,
                max_novel_proposals=limits.max_novel_proposals,
            )
            if last_evaluation.accepted:
                return AnalysisProposalRoutingResult(
                    proposals=proposals,
                    evaluation=last_evaluation,
                    router_evaluation=last_evaluation,
                    used_fallback=False,
                    analysis_inconclusive=False,
                    router_calls=calls,
                    router_retries=max(0, calls - 1),
                    reserved_tokens=calls * token_cost,
                    raw_responses=tuple(raw_responses),
                    router_call_timestamps=tuple([*historical_timestamps, *([now_epoch] * calls)]),
                )
            reasons = ", ".join(item.reason for item in last_evaluation.rejected)
            errors.append(f"all router proposals rejected: {reasons or 'no accepted proposals'}")
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}"[:1000])

    fallback_count = max(1, min(desired, limits.max_novel_proposals or 1))
    fallback_values = (
        fallback_planner(context, fallback_count)
        if fallback_planner is not None
        else _deterministic_analysis_proposals(context, objective_terms, fallback_count)
    )
    fallback_proposals: list[AnalysisProposal] = []
    for item in fallback_values:
        if isinstance(item, AnalysisProposal):
            fallback_proposals.append(item)
        elif isinstance(item, PlanBranch):
            fallback_proposals.append(
                AnalysisProposal(item, 1.0, 1.0, tuple(objective_terms) or ("unresolved objective",))
            )
        elif isinstance(item, Mapping) and "branch" in item:
            fallback_proposals.append(AnalysisProposal.from_dict(item))
        else:
            branch = PlanBranch.from_dict(item)
            fallback_proposals.append(
                AnalysisProposal(branch, 1.0, 1.0, tuple(objective_terms) or ("unresolved objective",))
            )
    if not fallback_proposals:
        raise TaskProposalRouterError(
            "analysis router was inconclusive and deterministic fallback returned no proposals"
        )
    fallback_evaluation = evaluate_analysis_proposals(
        fallback_proposals,
        objective_terms=objective_terms,
        known_proposal_ids=known_proposal_ids,
        min_confidence=0.0,
        min_novelty=0.0,
        max_novel_proposals=limits.max_novel_proposals,
    )
    combined_evaluation = AnalysisProposalEvaluation(
        accepted=fallback_evaluation.accepted,
        rejected=tuple([*last_evaluation.rejected, *fallback_evaluation.rejected]),
        plan_evaluation=fallback_evaluation.plan_evaluation,
    )
    return AnalysisProposalRoutingResult(
        proposals=tuple(fallback_proposals),
        evaluation=combined_evaluation,
        router_evaluation=last_evaluation,
        used_fallback=True,
        analysis_inconclusive=True,
        router_calls=calls,
        router_retries=max(0, calls - 1),
        reserved_tokens=calls * token_cost,
        router_error="; ".join(errors) or limit_reason or "router was not called",
        raw_responses=tuple(raw_responses),
        router_call_timestamps=tuple([*historical_timestamps, *([now_epoch] * calls)]),
        limit_reason=limit_reason,
    )
