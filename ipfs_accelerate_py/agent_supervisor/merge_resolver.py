"""LLM merge-conflict resolver payloads for autonomous agent supervisors."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any, Sequence

from .event_log import read_jsonl_events


LLM_MERGE_RESOLVER_COMMAND_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_COMMAND"
LLM_MERGE_RESOLVER_TIMEOUT_ENV = "IPFS_ACCELERATE_AGENT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS"
DEFAULT_LLM_MERGE_RESOLVER_TIMEOUT_SECONDS = 600.0
DEFAULT_PROMPT_HEADING = "Resolve the autonomous-agent supervisor merge conflict in this repository."
DEFAULT_COMPLETION_RULE = "Do not unblock the source task until validation passes."


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
