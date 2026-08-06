#!/usr/bin/env python3
"""Non-interactive Claude Code / Gemini CLI implement runner (stdin prompt).

The implementation daemon streams the task prompt on stdin and expects a
subprocess argv that performs agent-style edits in the worktree cwd.  This
module adapts Claude Code and Gemini CLI to that contract without charging
quota at import time.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _resolve_claude() -> str:
    for env in (
        "IPFS_ACCELERATE_AGENT_CLAUDE_BIN",
        "CLAUDE_BIN",
        "ANTHROPIC_CLI_BIN",
    ):
        configured = str(os.environ.get(env) or "").strip()
        if configured:
            path = Path(configured).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
            found = shutil.which(configured.split()[0])
            if found:
                return found
    found = shutil.which("claude")
    if not found:
        raise SystemExit("claude CLI not found on PATH")
    return found


def _resolve_gemini_argv() -> list[str]:
    for env in (
        "IPFS_ACCELERATE_AGENT_GEMINI_BIN",
        "ipfs_accelerate_py_GEMINI_CLI_CMD",
        "GEMINI_BIN",
    ):
        configured = str(os.environ.get(env) or "").strip()
        if not configured:
            continue
        # Templates may include {prompt}; strip placeholders for argv base.
        token = (
            configured.replace("{prompt}", "")
            .replace("{model}", "")
            .strip()
        )
        parts = token.split()
        if not parts:
            continue
        if parts[0] in {"npx", "npm"}:
            if shutil.which(parts[0]):
                return parts
            continue
        path = Path(parts[0]).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return [str(path), *parts[1:]]
        found = shutil.which(parts[0])
        if found:
            return [found, *parts[1:]]
    if shutil.which("gemini"):
        return ["gemini"]
    if shutil.which("npx"):
        return ["npx", "--yes", "@google/gemini-cli"]
    raise SystemExit("gemini CLI not found on PATH")


def _run_claude(*, workspace: Path, model: str, prompt: str) -> int:
    claude = _resolve_claude()
    # Claude Code non-interactive: -p prints, --dangerously-skip-permissions
    # allows unattended tool use in the workspace. Prompt is the -p argument
    # (CLI does not reliably accept multi-MB prompts on bare stdin alone).
    command = [
        claude,
        "-p",
        prompt,
        "--dangerously-skip-permissions",
        "--output-format",
        "text",
    ]
    if model:
        command.extend(["--model", model])
    env = os.environ.copy()
    completed = subprocess.run(
        command,
        cwd=str(workspace),
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _run_gemini(*, workspace: Path, model: str, prompt: str) -> int:
    base = _resolve_gemini_argv()
    # Gemini CLI commonly accepts the prompt as a positional or via -p.
    command = list(base)
    if model:
        command.extend(["-m", model])
    # Prefer explicit prompt flag when using the official CLI shape.
    if any("gemini" in part for part in command):
        command.extend(["-p", prompt])
    else:
        command.append(prompt)
    # YOLO / auto-approve when available (ignore if unsupported).
    if "--yolo" not in command:
        command.append("--yolo")
    env = os.environ.copy()
    completed = subprocess.run(
        command,
        cwd=str(workspace),
        env=env,
        check=False,
    )
    return int(completed.returncode)


def _resolve_mistral() -> str:
    for env in (
        "IPFS_ACCELERATE_MISTRAL_VIBE_CLI_CMD",
        "ipfs_accelerate_py_MISTRAL_VIBE_CLI_CMD",
        "MISTRAL_VIBE_BIN",
        "VIBE_BIN",
    ):
        configured = str(os.environ.get(env) or "").strip()
        if configured:
            token = configured.split()[0]
            path = Path(token).expanduser()
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
            found = shutil.which(token)
            if found:
                return found
    for candidate in ("vibe", "mistral-vibe"):
        found = shutil.which(candidate)
        if found:
            return found
    raise SystemExit("mistral vibe CLI not found on PATH")


def _run_mistral(*, workspace: Path, model: str, prompt: str) -> int:
    vibe = _resolve_mistral()
    # Non-interactive prompt run; flags vary by vibe version.
    command = [
        vibe,
        "--prompt",
        prompt,
        "--output",
        "text",
        "--max-turns",
        "100",
    ]
    if model:
        command.extend(["--model", model])
    env = os.environ.copy()
    completed = subprocess.run(
        command,
        cwd=str(workspace),
        env=env,
        check=False,
    )
    return int(completed.returncode)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        required=True,
        choices=("claude", "gemini", "mistral"),
        help="CLI implement provider",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        type=Path,
        help="Worktree root (process cwd for the CLI)",
    )
    parser.add_argument(
        "--model",
        default="",
        help="Optional model override",
    )
    args = parser.parse_args(argv)
    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    prompt = sys.stdin.read()
    if not prompt.strip():
        print("empty implementation prompt on stdin", file=sys.stderr)
        return 2
    model = str(args.model or "").strip()
    if args.provider == "claude":
        return _run_claude(workspace=workspace, model=model, prompt=prompt)
    if args.provider == "mistral":
        return _run_mistral(workspace=workspace, model=model, prompt=prompt)
    return _run_gemini(workspace=workspace, model=model, prompt=prompt)


if __name__ == "__main__":
    raise SystemExit(main())
