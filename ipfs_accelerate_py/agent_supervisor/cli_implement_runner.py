#!/usr/bin/env python3
"""Non-interactive CLI implement runner (stdin prompt).

The implementation daemon streams the task prompt on stdin and expects a
subprocess argv that performs agent-style edits in the worktree cwd.  This
module adapts Claude Code, Gemini CLI, Mistral Vibe, and Muse Code to that
contract without charging quota at import time. Muse Code may install the
official ``muse`` binary when explicitly selected.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _operator_home_dir() -> Path:
    try:
        import pwd

        return Path(pwd.getpwuid(os.getuid()).pw_dir)
    except Exception:
        return Path.home()


def _host_cli_binary(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    for path in (
        _operator_home_dir() / ".local" / "bin" / name,
        Path("/usr/local/bin") / name,
        Path("/usr/bin") / name,
        Path("/opt/cli-bin") / name,
    ):
        try:
            if path.is_file() and os.access(path, os.X_OK):
                return str(path)
        except OSError:
            continue
    return None


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
    found = _host_cli_binary("claude")
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
    gemini = _host_cli_binary("gemini")
    if gemini:
        return [gemini]
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
        found = _host_cli_binary(candidate)
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


def _resolve_muse(*, auto_install: bool = True) -> str:
    try:
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.cli_provider_balance import (
            ensure_muse_cli_binary,
            resolve_muse_cli_binary,
        )
    except Exception:
        ensure_muse_cli_binary = None  # type: ignore[assignment]
        resolve_muse_cli_binary = None  # type: ignore[assignment]
    if auto_install and callable(ensure_muse_cli_binary):
        found = ensure_muse_cli_binary()
        if found:
            return found
    if callable(resolve_muse_cli_binary):
        found = resolve_muse_cli_binary()
        if found:
            return found
    try:
        from ipfs_accelerate_py.cli_runtime.installers.muse import ensure_muse

        result = ensure_muse(auto_install=auto_install)
        if result.available and result.executable:
            return str(result.executable)
    except Exception:
        pass
    found = shutil.which("muse")
    if found:
        return found
    default = Path.home() / ".local" / "bin" / "muse"
    if default.is_file() and os.access(default, os.X_OK):
        return str(default)
    raise SystemExit("muse CLI not found on PATH; official install failed")


def _run_muse(*, workspace: Path, model: str, prompt: str) -> int:
    muse = _resolve_muse(auto_install=True)
    try:
        from ipfs_accelerate_py.cli_runtime.providers.muse import (
            DEFAULT_AGENT_MAX_MODEL_STEPS,
            build_muse_exec_argv,
            build_muse_process_env,
        )
    except Exception as exc:
        print(f"muse adapter unavailable: {exc}", file=sys.stderr)
        return 2
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="muse-supervisor-prompt-",
        suffix=".txt",
        delete=False,
    )
    prompt_file = handle.name
    try:
        handle.write(prompt)
        handle.flush()
        handle.close()
        argv = build_muse_exec_argv(
            executable=muse,
            prompt_file=prompt_file,
            model_name=model or None,
            json_events=True,
            disable_approval=True,
            yolo=False,
            workspace=str(workspace),
            max_model_steps=DEFAULT_AGENT_MAX_MODEL_STEPS,
        )
        overlay = build_muse_process_env()
        env = os.environ.copy()
        for key, value in overlay.items():
            if value is None:
                env.pop(str(key), None)
            else:
                env[str(key)] = str(value)
        completed = subprocess.run(
            argv,
            cwd=str(workspace),
            env=env,
            check=False,
        )
        return int(completed.returncode)
    finally:
        try:
            os.unlink(prompt_file)
        except OSError:
            pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider",
        required=True,
        choices=("claude", "gemini", "mistral", "muse_code"),
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
    if args.provider == "muse_code":
        return _run_muse(workspace=workspace, model=model, prompt=prompt)
    return _run_gemini(workspace=workspace, model=model, prompt=prompt)


if __name__ == "__main__":
    raise SystemExit(main())
