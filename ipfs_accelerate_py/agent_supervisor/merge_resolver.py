"""LLM merge-conflict resolver payloads for autonomous agent supervisors."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

from .event_log import read_jsonl_events


LLM_MERGE_RESOLVER_COMMAND_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND"
LLM_MERGE_RESOLVER_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS"
DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS = 600.0
DEFAULT_PROMPT_HEADING = "Resolve the autonomous-agent supervisor merge conflict in this repository."
DEFAULT_COMPLETION_RULE = "Do not unblock the source task until validation passes."
MergePromptCallback = Callable[..., str]
MergeResolverPayloadCallback = Callable[..., dict[str, Any]]
MergeResolverInvoker = Callable[..., dict[str, Any]]


@dataclass(frozen=True)
class MergeResolverCliConfig:
    """Project-specific defaults for the reusable merge-resolver CLI."""

    default_events_path: Path
    default_repo_root: Path
    prompt_heading: str = DEFAULT_PROMPT_HEADING
    completion_rule: str = DEFAULT_COMPLETION_RULE
    extra_rules: Sequence[str] = field(default_factory=tuple)
    primary_command_env_var: str = ""
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV
    description: str = "Build or invoke an LLM merge resolver for agent-supervisor events"
    missing_event_exit_code: int = 0
    apply_failed_exit_code: int = 1


@dataclass(frozen=True)
class ConfiguredMergeResolverRunner:
    """Project-bound runner wiring for a merge-resolver CLI."""

    config: MergeResolverCliConfig

    def parse_args(self, argv: Sequence[str] | None = None) -> argparse.Namespace:
        """Parse merge-resolver CLI args using the bound config."""

        return build_configured_merge_resolver_arg_parser(self.config).parse_args(argv)

    def run(self, argv: Sequence[str] | None = None) -> int:
        """Run the configured merge-resolver CLI."""

        return run_configured_merge_resolver_cli(self.config, argv)

    def build_merge_prompt(self) -> MergePromptCallback:
        """Build a prompt callback using the bound project wording."""

        return build_merge_prompt_callback(
            prompt_heading=self.config.prompt_heading,
            completion_rule=self.config.completion_rule,
            extra_rules=self.config.extra_rules,
        )

    def resolver_payload(self) -> MergeResolverPayloadCallback:
        """Build a resolver payload callback using the bound project wording."""

        return build_resolver_payload_callback(
            prompt_heading=self.config.prompt_heading,
            completion_rule=self.config.completion_rule,
            extra_rules=self.config.extra_rules,
        )

    def llm_resolver_invoker(self) -> MergeResolverInvoker:
        """Build an LLM resolver invoker using the bound command env vars."""

        return build_llm_merge_resolver_invoker(
            primary_command_env_var=self.config.primary_command_env_var,
            fallback_command_env_var=self.config.fallback_command_env_var,
        )


def build_configured_merge_resolver_runner(
    *,
    default_events_path: Path | str,
    default_repo_root: Path | str,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] = (),
    primary_command_env_var: str = "",
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV,
    description: str = "Build or invoke an LLM merge resolver for agent-supervisor events",
    missing_event_exit_code: int = 0,
    apply_failed_exit_code: int = 1,
) -> ConfiguredMergeResolverRunner:
    """Build reusable merge-resolver runner wiring bound to project inputs."""

    return ConfiguredMergeResolverRunner(
        MergeResolverCliConfig(
            default_events_path=Path(default_events_path),
            default_repo_root=Path(default_repo_root),
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=tuple(extra_rules),
            primary_command_env_var=primary_command_env_var,
            fallback_command_env_var=fallback_command_env_var,
            description=description,
            missing_event_exit_code=missing_event_exit_code,
            apply_failed_exit_code=apply_failed_exit_code,
        )
    )


def build_namespace_merge_resolver_runner(
    *,
    repo_root: Path | str,
    namespace: str,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] = (),
    state_prefix: str | None = None,
    env_prefix: str = "",
    state_dir: Path | str | None = None,
    description: str = "Build or invoke an LLM merge resolver for agent-supervisor events",
    missing_event_exit_code: int = 0,
    apply_failed_exit_code: int = 1,
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV,
) -> ConfiguredMergeResolverRunner:
    """Build a merge-resolver runner using the standard namespace state layout."""

    from .implementation_daemon_runner import namespace_implementation_state_artifact_paths
    from .wrapper_utils import agent_supervisor_namespace_paths, prefixed_env_var

    resolved_repo_root = Path(repo_root)
    namespace_paths = agent_supervisor_namespace_paths(resolved_repo_root, namespace)
    state_paths = namespace_implementation_state_artifact_paths(
        namespace_paths,
        state_prefix=state_prefix,
        state_dir=state_dir,
    )
    return build_configured_merge_resolver_runner(
        default_events_path=state_paths["events_path"],
        default_repo_root=resolved_repo_root,
        prompt_heading=prompt_heading,
        completion_rule=completion_rule,
        extra_rules=extra_rules,
        primary_command_env_var=(
            prefixed_env_var(env_prefix, "LLM_MERGE_RESOLVER_COMMAND") if env_prefix else ""
        ),
        fallback_command_env_var=fallback_command_env_var,
        description=description,
        missing_event_exit_code=missing_event_exit_code,
        apply_failed_exit_code=apply_failed_exit_code,
    )


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    return read_jsonl_events(path, repair=True)


def latest_failed_merge_event(events: list[dict[str, Any]], *, task_id: str | None = None) -> dict[str, Any] | None:
    """Return the newest merge failure event, optionally filtered by task id."""

    for event in reversed(events):
        if str(event.get("type") or "") not in {"implementation_finished", "merge_finished", "merge_reconciled"}:
            continue
        if task_id and str(event.get("task_id") or "") != task_id:
            continue
        merge_result = event.get("merge_result") if isinstance(event.get("merge_result"), dict) else event
        if not isinstance(merge_result, dict):
            continue
        if not merge_result.get("attempted") or merge_result.get("merged"):
            continue
        if str(merge_result.get("reason") or "") == "not_attempted":
            continue
        return event
    return None


def unmerged_paths(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "diff", "--name-only", "--diff-filter=U"],
        cwd=repo_root,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return sorted(line.strip() for line in result.stdout.splitlines() if line.strip())


def compact_text(value: Any, *, limit: int = 2000) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[:limit] + "\n...[truncated]"


def resolver_timeout_seconds(value: str | float | int | None = None) -> float | None:
    raw_value = os.environ.get(LLM_MERGE_RESOLVER_TIMEOUT_ENV, "") if value is None else value
    if raw_value in {None, ""}:
        return DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS
    try:
        timeout_seconds = float(raw_value)
    except (TypeError, ValueError):
        return DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS
    if timeout_seconds <= 0:
        return None
    return timeout_seconds


def _merge_result(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("merge_result")
    return value if isinstance(value, dict) else event


def build_merge_prompt(
    *,
    event: dict[str, Any],
    repo_root: Path,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> str:
    """Build an LLM prompt that can resolve a semantic merge conflict."""

    merge_result = _merge_result(event)
    command = merge_result.get("command") or []
    if isinstance(command, list):
        command_text = shlex.join(str(part) for part in command)
    else:
        command_text = str(command)
    paths = unmerged_paths(repo_root)
    dirty_paths = merge_result.get("dirty_paths") or []
    rules = [
        "Inspect the conflicted files and implementation branch before editing.",
        "Preserve the semantic intent of both sides when possible.",
        "Keep changes scoped to the task and conflict resolution.",
        "Run the task validation after resolving the conflict.",
        "Commit the merge resolution in the owning repository or submodule.",
        completion_rule,
    ]
    if extra_rules:
        rules.extend(str(rule) for rule in extra_rules if str(rule).strip())
    return "\n".join(
        [
            prompt_heading,
            "",
            f"Task id: {event.get('task_id') or merge_result.get('task_id')}",
            f"Attempt: {event.get('attempt') or merge_result.get('attempt')}",
            f"Implementation branch: {merge_result.get('branch')}",
            f"Target branch: {merge_result.get('target_branch')}",
            f"Merge reason: {merge_result.get('reason')}",
            f"Merge command: {command_text}",
            f"Repository: {repo_root}",
            f"Unmerged paths: {', '.join(paths) or 'none reported by git'}",
            f"Dirty paths: {', '.join(str(item) for item in dirty_paths) or 'none recorded'}",
            "",
            "Rules:",
            *(f"{index}. {rule}" for index, rule in enumerate(rules, start=1)),
            "",
            "Merge stdout excerpt:",
            compact_text(merge_result.get("stdout")),
            "",
            "Merge stderr excerpt:",
            compact_text(merge_result.get("stderr")),
        ]
    )


def build_merge_prompt_callback(
    *,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> MergePromptCallback:
    """Build a prompt callback with project-specific merge-resolution wording."""

    configured_extra_rules = tuple(extra_rules or ())

    def callback(*, event: dict[str, Any], repo_root: Path) -> str:
        return build_merge_prompt(
            event=event,
            repo_root=repo_root,
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=configured_extra_rules,
        )

    return callback


def resolver_payload(
    *,
    events_path: Path,
    repo_root: Path,
    task_id: str | None = None,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Return a dry-run JSON payload for the latest merge failure."""

    event = latest_failed_merge_event(iter_jsonl(events_path), task_id=task_id)
    if event is None:
        return {
            "found": False,
            "task_id": task_id,
            "events_path": str(events_path),
            "repo_root": str(repo_root),
            "prompt": "",
        }
    merge_result = _merge_result(event)
    return {
        "found": True,
        "task_id": str(event.get("task_id") or merge_result.get("task_id") or ""),
        "attempt": event.get("attempt") or merge_result.get("attempt"),
        "events_path": str(events_path),
        "repo_root": str(repo_root),
        "branch": str(merge_result.get("branch") or ""),
        "target_branch": str(merge_result.get("target_branch") or ""),
        "command": merge_result.get("command") or [],
        "reason": str(merge_result.get("reason") or ""),
        "dirty_paths": merge_result.get("dirty_paths") or [],
        "unmerged_paths": unmerged_paths(repo_root),
        "prompt": build_merge_prompt(
            event=event,
            repo_root=repo_root,
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=extra_rules,
        ),
    }


def build_resolver_payload_callback(
    *,
    prompt_heading: str = DEFAULT_PROMPT_HEADING,
    completion_rule: str = DEFAULT_COMPLETION_RULE,
    extra_rules: Sequence[str] | None = None,
) -> MergeResolverPayloadCallback:
    """Build a resolver-payload callback with project-specific prompt defaults."""

    configured_extra_rules = tuple(extra_rules or ())

    def callback(*, events_path: Path, repo_root: Path, task_id: str | None = None) -> dict[str, Any]:
        return resolver_payload(
            events_path=events_path,
            repo_root=repo_root,
            task_id=task_id,
            prompt_heading=prompt_heading,
            completion_rule=completion_rule,
            extra_rules=configured_extra_rules,
        )

    return callback


def invoke_llm_resolver(
    payload: dict[str, Any],
    *,
    command_template: str | None = None,
    timeout_seconds: float | None = None,
) -> dict[str, Any]:
    """Invoke an external LLM resolver command with the prompt on stdin."""

    command_template = (command_template or os.environ.get(LLM_MERGE_RESOLVER_COMMAND_ENV, "")).strip()
    if not command_template:
        return {
            **payload,
            "applied": False,
            "apply_error": f"{LLM_MERGE_RESOLVER_COMMAND_ENV} is not set",
        }
    command = shlex.split(command_template)
    timeout = resolver_timeout_seconds(timeout_seconds)
    try:
        result = subprocess.run(
            command,
            cwd=payload.get("repo_root") or None,
            input=str(payload.get("prompt") or ""),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            **payload,
            "applied": False,
            "llm_command": command,
            "llm_timeout": True,
            "llm_timeout_seconds": timeout,
            "llm_returncode": None,
            "llm_stdout": compact_text(exc.stdout),
            "llm_stderr": compact_text(exc.stderr),
            "apply_error": f"LLM merge resolver timed out after {timeout} seconds",
        }
    return {
        **payload,
        "applied": result.returncode == 0,
        "llm_command": command,
        "llm_timeout": False,
        "llm_timeout_seconds": timeout,
        "llm_returncode": result.returncode,
        "llm_stdout": compact_text(result.stdout),
        "llm_stderr": compact_text(result.stderr),
    }


def _configured_command_template(primary_env_var: str, fallback_env_var: str) -> str | None:
    for env_var in (primary_env_var, fallback_env_var):
        if not env_var:
            continue
        configured = os.environ.get(env_var, "").strip()
        if configured:
            return configured
    return None


def _missing_command_error(primary_env_var: str, fallback_env_var: str) -> str:
    env_vars = [env_var for env_var in (primary_env_var, fallback_env_var) if env_var]
    if not env_vars:
        return "LLM merge resolver command is not configured"
    return f"{' or '.join(env_vars)} is not set"


def build_llm_merge_resolver_invoker(
    *,
    primary_command_env_var: str = "",
    fallback_command_env_var: str = LLM_MERGE_RESOLVER_COMMAND_ENV,
    missing_command_error: str = "",
) -> MergeResolverInvoker:
    """Build an invoker that resolves project and fallback command env vars."""

    def callback(payload: dict[str, Any], *, timeout_seconds: float | None = None) -> dict[str, Any]:
        command_template = _configured_command_template(primary_command_env_var, fallback_command_env_var)
        if command_template is None:
            return {
                **payload,
                "applied": False,
                "apply_error": missing_command_error
                or _missing_command_error(primary_command_env_var, fallback_command_env_var),
            }
        return invoke_llm_resolver(payload, command_template=command_template, timeout_seconds=timeout_seconds)

    return callback


def build_configured_merge_resolver_arg_parser(config: MergeResolverCliConfig) -> argparse.ArgumentParser:
    """Build a standard parser for a configured merge-resolver wrapper."""

    parser = argparse.ArgumentParser(description=config.description)
    parser.add_argument("--task-id", default=None, help="Resolve the latest merge failure for this task id.")
    parser.add_argument("--events-path", type=Path, default=config.default_events_path)
    parser.add_argument("--repo-root", type=Path, default=config.default_repo_root)
    parser.add_argument("--apply", action="store_true", help="Invoke the configured LLM resolver command.")
    parser.add_argument("--command", default=None, help="Resolver command template. Defaults to configured env vars.")
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=f"Resolver subprocess timeout. Defaults to {LLM_MERGE_RESOLVER_TIMEOUT_ENV} or 600 seconds; <=0 disables.",
    )
    return parser


def run_configured_merge_resolver_cli(
    config: MergeResolverCliConfig,
    argv: Sequence[str] | None = None,
) -> int:
    """Run a project-configured merge-resolver dry-run/apply CLI."""

    args = build_configured_merge_resolver_arg_parser(config).parse_args(argv)
    payload = resolver_payload(
        events_path=args.events_path,
        repo_root=args.repo_root,
        task_id=args.task_id,
        prompt_heading=config.prompt_heading,
        completion_rule=config.completion_rule,
        extra_rules=config.extra_rules,
    )
    if args.apply and payload.get("found"):
        if args.command:
            payload = invoke_llm_resolver(payload, command_template=args.command, timeout_seconds=args.timeout_seconds)
        else:
            invoker = build_llm_merge_resolver_invoker(
                primary_command_env_var=config.primary_command_env_var,
                fallback_command_env_var=config.fallback_command_env_var,
            )
            payload = invoker(payload, timeout_seconds=args.timeout_seconds)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not payload.get("found"):
        return config.missing_event_exit_code
    if args.apply and not payload.get("applied"):
        return config.apply_failed_exit_code
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build or invoke an LLM merge resolver for agent-supervisor events")
    parser.add_argument("--events-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--task-id", default=None)
    parser.add_argument("--apply", action="store_true", help="Invoke the configured resolver command")
    parser.add_argument("--command", default=None, help="Resolver command template. Defaults to env var.")
    parser.add_argument("--prompt-heading", default=DEFAULT_PROMPT_HEADING)
    parser.add_argument("--completion-rule", default=DEFAULT_COMPLETION_RULE)
    parser.add_argument("--extra-rule", action="append", default=[])
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=None,
        help=f"Resolver subprocess timeout. Defaults to {LLM_MERGE_RESOLVER_TIMEOUT_ENV} or 600 seconds; <=0 disables.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    payload = resolver_payload(
        events_path=args.events_path,
        repo_root=args.repo_root.resolve(),
        task_id=args.task_id,
        prompt_heading=args.prompt_heading,
        completion_rule=args.completion_rule,
        extra_rules=args.extra_rule,
    )
    if args.apply:
        payload = invoke_llm_resolver(payload, command_template=args.command, timeout_seconds=args.timeout_seconds)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.apply and payload.get("found") and not payload.get("applied"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
