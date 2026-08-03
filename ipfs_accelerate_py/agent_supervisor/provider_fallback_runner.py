#!/usr/bin/env python3
"""Run one stdin-driven provider command with one ordered fallback.

The implementation daemon supplies commands as JSON argument vectors.  This
runner deliberately does not use a shell: both children receive the exact same
stdin prompt, inherit this process's stdout/stderr streams, and run in the same
resolved workspace.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def _command_from_json(value: str, *, field_name: str) -> list[str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if (
        not isinstance(payload, list)
        or not payload
        or any(not isinstance(item, str) or not item for item in payload)
    ):
        raise ValueError(f"{field_name} must be a non-empty JSON string array")
    return list(payload)


def _run_provider(
    command: Sequence[str],
    *,
    workspace: Path,
    prompt: str,
    provider_name: str,
) -> int | None:
    """Run one provider, returning ``None`` only when it cannot be launched."""

    try:
        completed = subprocess.run(
            list(command),
            cwd=workspace,
            input=prompt,
            text=True,
            check=False,
        )
    except OSError as exc:
        print(
            f"{provider_name} provider could not launch: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return None
    return int(completed.returncode)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a primary implementation provider with one fallback."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--primary-provider", required=True)
    parser.add_argument("--fallback-provider", required=True)
    parser.add_argument("--primary-command-json", required=True)
    parser.add_argument("--fallback-command-json", required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2

    try:
        primary_command = _command_from_json(
            args.primary_command_json,
            field_name="primary command",
        )
        fallback_command = _command_from_json(
            args.fallback_command_json,
            field_name="fallback command",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    prompt = sys.stdin.read()
    primary_provider = str(args.primary_provider).strip() or "primary"
    fallback_provider = str(args.fallback_provider).strip() or "fallback"

    os.chdir(workspace)
    primary_returncode = _run_provider(
        primary_command,
        workspace=workspace,
        prompt=prompt,
        provider_name=primary_provider,
    )
    if primary_returncode == 0:
        return 0

    if primary_returncode is not None:
        print(
            f"{primary_provider} provider exited with {primary_returncode}; "
            f"falling back to {fallback_provider}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"falling back to {fallback_provider}",
            file=sys.stderr,
            flush=True,
        )

    fallback_returncode = _run_provider(
        fallback_command,
        workspace=workspace,
        prompt=prompt,
        provider_name=fallback_provider,
    )
    return 127 if fallback_returncode is None else fallback_returncode


if __name__ == "__main__":
    raise SystemExit(main())
