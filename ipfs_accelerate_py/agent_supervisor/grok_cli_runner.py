#!/usr/bin/env python3
"""Thin process entry: Grok Build CLI agent for implementation worktrees.

Reads the implementation prompt from stdin (daemon contract), writes it to a
temp prompt file, then execs the official ``grok`` binary with agent-capable
flags. Command policy lives next to other CLI peers in :mod:`llm_router`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.agent_supervisor.todo_daemon.llm_defaults import (
    DEFAULT_CODEX_QUOTA_FALLBACK_MODEL,
    DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT,
    DEFAULT_GROK_PRIMARY_MODEL,
)

DEFAULT_GROK_MODEL = DEFAULT_GROK_PRIMARY_MODEL
# Grok CLI validates --max-turns as 1..=4294967295 (u32::MAX).
DEFAULT_GROK_MAX_TURNS = 4_294_967_295
MAX_CODEX_FALLBACK_ARGUMENTS = 64
MAX_CODEX_FALLBACK_ARGUMENT_BYTES = 4_096
GROK_QUOTA_TRANSCRIPT_TAIL_BYTES = 128 * 1024


def _resolve_grok_bin(configured: str = "") -> str:
    if configured.strip():
        path = Path(configured).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return str(path)
    try:
        from ipfs_accelerate_py.llm_router import _grok_cli_command

        candidate = str(_grok_cli_command() or "").strip()
        if candidate:
            found = shutil.which(candidate) or (
                candidate if Path(candidate).is_file() else ""
            )
            if found:
                return found
    except Exception:
        pass
    return shutil.which("grok") or ""


def build_grok_agent_command(
    *,
    workspace: Path,
    prompt_file: Path,
    model: str,
    max_turns: int,
    permission_mode: str,
    grok_bin: str,
) -> list[str]:
    """Build an agent-mode ``grok`` argv for an implementation worktree."""

    cmd = [
        grok_bin,
        "--cwd",
        str(workspace),
        "--model",
        model,
        "--permission-mode",
        permission_mode,
        "--always-approve",
        "--max-turns",
        str(max_turns),
        "--output-format",
        "plain",
        "--prompt-file",
        str(prompt_file),
    ]
    return cmd


def resolve_codex_quota_fallback_executable(
    *,
    workspace: str | Path,
    configured: str = "",
) -> str:
    """Resolve a Codex executable the Grok workspace cannot replace.

    Preference order: an absolute configured path, then ``PATH`` lookup.
    Workspace-relative or workspace-contained candidates are rejected so an
    implementation worktree cannot supply its own "codex" binary.
    """

    workspace_path = Path(workspace).expanduser().resolve()
    codex_candidate = str(configured or shutil.which("codex") or "").strip()
    if not codex_candidate:
        return ""
    candidate_path = Path(codex_candidate).expanduser()
    if not candidate_path.is_absolute():
        resolved_from_path = shutil.which(codex_candidate)
        if not resolved_from_path:
            return ""
        candidate_path = Path(resolved_from_path)
    try:
        resolved_candidate = candidate_path.resolve(strict=True)
    except OSError:
        return ""
    if (
        not resolved_candidate.is_file()
        or not os.access(resolved_candidate, os.X_OK)
        or resolved_candidate.name.casefold() not in {"codex", "codex.exe"}
        or resolved_candidate.is_relative_to(workspace_path)
        or candidate_path.expanduser().resolve(strict=False).is_relative_to(
            workspace_path
        )
    ):
        return ""
    return str(resolved_candidate)


def build_grok_quota_routed_agent_command(
    *,
    workspace: str | Path = ".",
    python_executable: str = "",
    grok_bin: str = "",
    codex_bin: str = "",
    max_turns: int = 100_000,
) -> list[str]:
    """Build the canonical Grok-4.5 then typed-quota Terra/medium route.

    The parent runner owns the Codex argv. Grok receives neither the
    executable/auth authority nor any way to invoke this fallback directly.
    """

    workspace_text = str(workspace)
    codex = resolve_codex_quota_fallback_executable(
        workspace=workspace,
        configured=codex_bin,
    )
    command = [
        str(python_executable or sys.executable),
        "-m",
        "ipfs_accelerate_py.agent_supervisor.grok_cli_runner",
        "--workspace",
        workspace_text,
        "--model",
        DEFAULT_GROK_MODEL,
        "--max-turns",
        str(max(1, int(max_turns))),
        "--mode",
        "agent",
    ]
    if codex:
        fallback = [
            codex,
            "exec",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "-s",
            "workspace-write",
            "-C",
            workspace_text,
            "-m",
            DEFAULT_CODEX_QUOTA_FALLBACK_MODEL,
            "-c",
            f'model_reasoning_effort="{DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT}"',
            "-",
        ]
        command.extend(
            [
                "--codex-fallback-command-json",
                json.dumps(fallback, separators=(",", ":")),
            ]
        )
    if str(grok_bin).strip():
        command.extend(["--grok-bin", str(grok_bin).strip()])
    return command


def _parse_codex_fallback_command(raw: str) -> list[str]:
    """Decode and pin the daemon-authored quota fallback without a shell."""

    if not raw.strip():
        return []
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("Codex fallback command is not valid JSON") from exc
    if not isinstance(payload, list) or not 2 <= len(payload) <= MAX_CODEX_FALLBACK_ARGUMENTS:
        raise ValueError("Codex fallback command must be a bounded argv array")
    command: list[str] = []
    for item in payload:
        if (
            not isinstance(item, str)
            or not item
            or len(item.encode("utf-8")) > MAX_CODEX_FALLBACK_ARGUMENT_BYTES
        ):
            raise ValueError(
                "Codex fallback command contains an invalid argument"
            )
        command.append(item)
    if Path(command[0]).name.lower() not in {"codex", "codex.exe"}:
        raise ValueError("Codex fallback executable must be codex")
    if command[1] != "exec" or command[-1] != "-":
        raise ValueError("Codex fallback command must use `codex exec ... -`")
    if command.count("--dangerously-bypass-approvals-and-sandbox") != 1:
        raise ValueError("Codex quota fallback must use the agent execution mode")

    index = 2
    config_values: list[str] = []
    while index < len(command) - 1:
        option = command[index]
        if option == "--dangerously-bypass-approvals-and-sandbox":
            index += 1
            continue
        if option not in {"-C", "-m", "-c"} or index + 1 >= len(command) - 1:
            raise ValueError("Codex quota fallback contains an unauthorized option")
        value = command[index + 1]
        if option == "-c":
            config_values.append(value)
        index += 2

    workspaces = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "-C"
    ]
    if len(workspaces) != 1:
        raise ValueError("Codex quota fallback must bind exactly one workspace")
    models = [
        command[index + 1]
        for index, item in enumerate(command[:-1])
        if item == "-m"
    ]
    if models != [DEFAULT_CODEX_QUOTA_FALLBACK_MODEL]:
        raise ValueError(
            "Codex quota fallback model must be "
            f"{DEFAULT_CODEX_QUOTA_FALLBACK_MODEL}"
        )
    allowed_config = re.compile(
        r'(?:model_context_window=[1-9][0-9]*|'
        r'model_reasoning_effort="medium"|'
        r'agents\.max_(?:threads|depth)=[1-9][0-9]*)'
    )
    if any(not allowed_config.fullmatch(value) for value in config_values):
        raise ValueError("Codex quota fallback contains an unauthorized config")
    expected_reasoning = (
        'model_reasoning_effort="'
        f"{DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT}"
        '"'
    )
    reasoning_values = [
        value
        for value in config_values
        if value.startswith("model_reasoning_effort=")
    ]
    if reasoning_values != [expected_reasoning]:
        raise ValueError(
            "Codex quota fallback reasoning effort must be "
            f"{DEFAULT_CODEX_QUOTA_FALLBACK_REASONING_EFFORT}"
        )
    return command


def _resolved_executable_path(executable: str) -> Path | None:
    """Resolve one executable to its real on-disk identity."""

    raw = str(executable or "").strip()
    if not raw:
        return None
    candidate = (
        shutil.which(raw)
        if os.sep not in raw and (os.altsep is None or os.altsep not in raw)
        else raw
    )
    if not candidate:
        return None
    try:
        # The discovery functions supplying the comparison identity already
        # require an executable file.  strict=False keeps this pure comparison
        # helper testable with injected discovery paths while still resolving
        # real symlink identities in production.
        return Path(candidate).expanduser().resolve(strict=False)
    except (OSError, RuntimeError):
        return None


def _same_executable_identity(configured: str, discovered: str) -> bool:
    configured_path = _resolved_executable_path(configured)
    discovered_path = _resolved_executable_path(discovered)
    return configured_path is not None and configured_path == discovered_path


def _grok_quota_exhausted(transcript: str) -> bool:
    """Recognize only explicit Grok account-quota exhaustion evidence.

    Generic nonzero exits, HTTP 429/rate limiting, authentication failures,
    network failures, and model/tool errors intentionally do not qualify.  The
    observed Grok Build account-exhaustion response binds an HTTP 402 status to
    the provider-specific usage-balance message.  A typed ``insufficient_quota``
    code is also accepted only when the same bounded diagnostic names Grok/xAI.
    """

    bounded = transcript[-GROK_QUOTA_TRANSCRIPT_TAIL_BYTES:]
    # Evidence must be self-contained in one provider diagnostic. Independent
    # searches over the whole tail would let unrelated stderr lines compose a
    # false quota authorization.
    for diagnostic in bounded.splitlines() or [bounded]:
        provider_named = re.search(
            r"\b(?:grok(?:\s+build)?|xai)\b",
            diagnostic,
            re.I,
        )
        if not provider_named:
            continue
        usage_exhausted = re.search(
            r"\b(?:usage\s+balance|account\s+quota|quota)\s+exhausted\b",
            diagnostic,
            re.I,
        )
        payment_required = re.search(
            r"(?:\b402\s+payment\s+required\b|"
            r'"http_status"\s*:\s*402\b|'
            r"\bstatus\s+402\b)",
            diagnostic,
            re.I,
        )
        if usage_exhausted and payment_required:
            return True
        if (
            re.search(
                r'"(?:code|type)"\s*:\s*"insufficient_quota"',
                diagnostic,
                re.I,
            )
            and re.search(r"\b(?:quota|usage)\b", diagnostic, re.I)
        ):
            return True
    return False


def _git_repository_effect_identity(
    workspace: Path,
) -> tuple[str, bytes] | None:
    """Return the exact HEAD and index/worktree state used by the fallback fence.

    An empty porcelain status alone is insufficient: Grok may commit its edits
    before reporting quota exhaustion, leaving the checkout clean.  Binding the
    resolved commit as well as the byte-exact status prevents Terra from
    following either committed or uncommitted Grok effects in the same attempt.
    """

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--verify", "HEAD^{commit}"],
            cwd=workspace,
            capture_output=True,
            check=False,
        )
        status = subprocess.run(
            [
                "git",
                "status",
                "--porcelain=v2",
                "--untracked-files=all",
                "--ignored=matching",
                "--ignore-submodules=none",
                "-z",
            ],
            cwd=workspace,
            capture_output=True,
            check=False,
        )
    except OSError:
        return None
    if int(head.returncode) != 0 or int(status.returncode) != 0:
        return None
    try:
        head_oid = bytes(head.stdout or b"").strip().decode("ascii", errors="strict")
    except UnicodeDecodeError:
        return None
    if not re.fullmatch(r"[0-9a-fA-F]{40,64}", head_oid):
        return None
    return head_oid.lower(), bytes(status.stdout or b"")


def _write_stream_chunk(target: object, chunk: bytes) -> None:
    target_buffer = getattr(target, "buffer", None)
    if target_buffer is not None:
        target_buffer.write(chunk)
    else:
        target.write(chunk.decode("utf-8", errors="replace"))
    flush = getattr(target_buffer or target, "flush", None)
    if callable(flush):
        flush()


def _run_grok_streaming(
    command: Sequence[str],
    *,
    env: dict[str, str],
) -> tuple[int, str]:
    """Stream Grok output live while retaining only a bounded diagnostic tail."""

    process = subprocess.Popen(
        list(command),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    tails = {"stdout": bytearray(), "stderr": bytearray()}
    pump_errors: list[BaseException] = []

    def pump(name: str, pipe: object, target: object) -> None:
        try:
            read_chunk = getattr(pipe, "read1", None)
            if not callable(read_chunk):
                read_chunk = pipe.read
            while True:
                # BufferedReader.read(size) may wait for the full size or EOF.
                # read1() returns bytes currently available from the child, so
                # short progress messages reach the supervisor's stall clock.
                chunk = read_chunk(64 * 1024)
                if not chunk:
                    break
                _write_stream_chunk(target, chunk)
                tail = tails[name]
                tail.extend(chunk)
                overflow = len(tail) - GROK_QUOTA_TRANSCRIPT_TAIL_BYTES
                if overflow > 0:
                    del tail[:overflow]
        except BaseException as exc:  # pragma: no cover - terminal failure
            pump_errors.append(exc)
        finally:
            close = getattr(pipe, "close", None)
            if callable(close):
                close()

    stdout_thread = threading.Thread(
        target=pump,
        args=("stdout", process.stdout, sys.stdout),
        name="grok-cli-stdout",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=pump,
        args=("stderr", process.stderr, sys.stderr),
        name="grok-cli-stderr",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    returncode = int(process.wait())
    stdout_thread.join()
    stderr_thread.join()
    if pump_errors:
        raise RuntimeError("unable to stream Grok CLI output") from pump_errors[0]
    # Only the provider process's stderr is eligible quota evidence.  Model
    # response text is stdout and cannot manufacture fallback authority.
    transcript = bytes(tails["stderr"]).decode(
        "utf-8",
        errors="replace",
    )
    return returncode, transcript


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Authorized Grok CLI agent entry (llm_router.grok_cli)."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--grok-bin", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--max-turns", default="")
    parser.add_argument(
        "--permission-mode",
        default="",
        help="Grok permission mode (default: bypassPermissions in agent mode).",
    )
    parser.add_argument(
        "--mode",
        default="agent",
        choices=("agent", "chat"),
        help="agent enables tool approvals for implementation work",
    )
    parser.add_argument(
        "--codex-fallback-command-json",
        default="",
        help=(
            "Internal GPT-5.6 Terra/medium argv. It is run at most once, and "
            "only after explicit pre-effect Grok quota exhaustion."
        ),
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        codex_fallback_command = _parse_codex_fallback_command(
            str(args.codex_fallback_command_json)
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    from ipfs_accelerate_py.llm_router import (
        LLMRouterError,
        build_grok_cli_command,
        build_grok_cli_env,
        find_grok_cli,
    )

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    if codex_fallback_command:
        workspace_indexes = [
            index
            for index, item in enumerate(codex_fallback_command[:-1])
            if item == "-C"
        ]
        if (
            len(workspace_indexes) != 1
            or Path(
                codex_fallback_command[workspace_indexes[0] + 1]
            ).expanduser().resolve()
            != workspace
        ):
            print(
                "Codex quota fallback workspace must match the Grok workspace",
                file=sys.stderr,
            )
            return 2

    grok_bin = str(args.grok_bin).strip() or find_grok_cli() or ""
    if not grok_bin:
        print("grok CLI not found on PATH", file=sys.stderr)
        return 127
    if codex_fallback_command:
        discovered_grok_bin = find_grok_cli() or ""
        if not _same_executable_identity(grok_bin, discovered_grok_bin):
            print(
                "Grok quota fallback executable does not match the "
                "supervisor-discovered Grok CLI",
                file=sys.stderr,
            )
            return 2
        discovered_codex_bin = shutil.which("codex") or ""
        if not _same_executable_identity(
            codex_fallback_command[0],
            discovered_codex_bin,
        ):
            print(
                "Codex quota fallback executable does not match the "
                "supervisor-discovered Codex CLI",
                file=sys.stderr,
            )
            return 2
        if str(args.mode) != "agent":
            print("Codex quota fallback requires Grok agent mode", file=sys.stderr)
            return 2

    model = (
        str(args.model).strip()
        or os.environ.get("IPFS_ACCELERATE_AGENT_GROK_MODEL", "").strip()
        or os.environ.get("ipfs_accelerate_py_GROK_CLI_MODEL", "").strip()
        or os.environ.get("GROK_CLI_MODEL", "").strip()
        or DEFAULT_GROK_MODEL
    )
    if codex_fallback_command and model != DEFAULT_GROK_MODEL:
        print(
            "Codex quota fallback requires Grok primary model "
            f"{DEFAULT_GROK_MODEL}",
            file=sys.stderr,
        )
        return 2
    max_turns_raw = (
        str(args.max_turns).strip()
        or os.environ.get("IPFS_ACCELERATE_AGENT_GROK_MAX_TURNS", "").strip()
        or os.environ.get("ipfs_accelerate_py_GROK_CLI_MAX_TURNS", "").strip()
        or str(DEFAULT_GROK_MAX_TURNS)
    )
    try:
        max_turns = max(1, min(DEFAULT_GROK_MAX_TURNS, int(max_turns_raw)))
    except ValueError:
        max_turns = DEFAULT_GROK_MAX_TURNS
    permission_mode = (
        str(args.permission_mode).strip()
        or os.environ.get("IPFS_ACCELERATE_AGENT_GROK_PERMISSION_MODE", "").strip()
        or os.environ.get(
            "ipfs_accelerate_py_GROK_CLI_PERMISSION_MODE", ""
        ).strip()
        or "bypassPermissions"
    )

    prompt = sys.stdin.read()
    if not prompt.strip():
        print("empty implementation prompt on stdin", file=sys.stderr)
        return 2

    prompt_path = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix="asref-grok-prompt-",
            suffix=".txt",
            delete=False,
        ) as handle:
            handle.write(prompt)
            prompt_path = handle.name

        try:
            cmd = build_grok_cli_command(
                mode=str(args.mode),
                workspace=workspace,
                model_name=model,
                max_turns=max_turns,
                grok_bin=grok_bin,
                prompt_file=prompt_path,
                permission_mode=permission_mode,
            )
            env = build_grok_cli_env()
        except LLMRouterError as exc:
            print(str(exc), file=sys.stderr)
            return 2

        os.chdir(workspace)
        repository_identity_before_grok = (
            _git_repository_effect_identity(workspace)
            if codex_fallback_command
            else None
        )
        primary_returncode, transcript = _run_grok_streaming(cmd, env=env)
        if primary_returncode == 0 or not codex_fallback_command:
            return primary_returncode

        if not _grok_quota_exhausted(transcript):
            print(
                "grok CLI failed without confirmed quota exhaustion; "
                "Codex fallback is forbidden",
                file=sys.stderr,
            )
            return primary_returncode
        repository_identity_after_grok = _git_repository_effect_identity(
            workspace
        )
        if (
            repository_identity_before_grok is None
            or repository_identity_before_grok[1]
            or repository_identity_after_grok
            != repository_identity_before_grok
        ):
            print(
                "grok quota exhausted after an unclean or changed workspace; "
                "cross-provider fallback is forbidden",
                file=sys.stderr,
            )
            return primary_returncode

        print(
            "grok CLI quota exhausted before repository effects; falling back "
            "once to codex gpt-5.6-terra with medium reasoning",
            file=sys.stderr,
        )
        try:
            fallback = subprocess.run(
                codex_fallback_command,
                cwd=workspace,
                env=os.environ.copy(),
                input=prompt,
                text=True,
                check=False,
            )
        except OSError as exc:
            print(f"unable to launch Codex fallback: {exc}", file=sys.stderr)
            return 127
        return int(fallback.returncode)
    finally:
        if prompt_path:
            try:
                os.unlink(prompt_path)
            except OSError:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
