#!/usr/bin/env python3
"""Thin process adapter for llm_router's explicit Grok -> Codex agent route.

The router owns the fixed provider order, failure vocabulary, policy predicate,
workspace side-effect gate, and route-record schema. This executable owns only
stdin/workspace fidelity, bounded private-safe stream replay, and child process
adaptation. Generic ``llm_router.generate_text`` fallback remains disabled for
side-effecting requests.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(_PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(_PACKAGE_ROOT))

from ipfs_accelerate_py.llm_router import (  # noqa: E402
    AGENT_CLI_PROVIDER_ROUTE_SCHEMA,
    AGENT_CLI_STDERR_LINE_LIMIT,
    GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY,
    AgentCLIActivityState,
    AgentCLIFailureClassification,
    AgentCLIProviderFailureKind,
    AgentCLIProviderResult,
    AgentCLIStderrSanitizer,
    LLMRouterError,
    classify_grok_agent_cli_failure,
    probe_grok_codex_agent_route_readiness,
    route_agent_cli_failure,
    safe_agent_cli_provider_label,
    serialize_agent_cli_route_record,
    snapshot_agent_cli_workspace,
)
from ipfs_accelerate_py.agent_supervisor.grok_cli_runner import (  # noqa: E402
    TRUSTED_FAILURE_RECEIPT_FD_ENV,
)

AGENT_ROUTE_POLICY = GROK_QUOTA_AUTH_OR_UNAVAILABLE_AGENT_ROUTE_POLICY

# Compatibility exports for callers that imported the former runner-local
# diagnostic helpers. Their implementation and policy now live in llm_router.
ProviderRunResult = AgentCLIProviderResult
GrokFailureKind = AgentCLIProviderFailureKind
GrokFailureClassification = AgentCLIFailureClassification
classify_grok_failure = classify_grok_agent_cli_failure
_ProviderStderrSanitizer = AgentCLIStderrSanitizer
_PROVIDER_STDERR_LINE_LIMIT = AGENT_CLI_STDERR_LINE_LIMIT
_PROVIDER_ROUTE_SCHEMA = AGENT_CLI_PROVIDER_ROUTE_SCHEMA


@dataclass(frozen=True)
class _ProviderExecution:
    result: AgentCLIProviderResult
    trusted_failure_receipt: str = ""


def _command_from_json(
    value: str,
    *,
    field_name: str,
    allow_empty: bool = False,
) -> list[str]:
    try:
        payload = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{field_name} must be valid JSON") from exc
    if (
        not isinstance(payload, list)
        or (not payload and not allow_empty)
        or any(not isinstance(item, str) or not item for item in payload)
    ):
        suffix = "JSON string array" if allow_empty else "non-empty JSON string array"
        raise ValueError(f"{field_name} must be a {suffix}")
    return list(payload)


def _uses_packaged_grok_adapter(command: Sequence[str]) -> bool:
    if len(command) < 2:
        return False
    expected = Path(__file__).with_name("grok_cli_runner.py").resolve()
    try:
        return Path(command[1]).expanduser().resolve() == expected
    except OSError:
        return False


def _read_private_failure_receipt(descriptor: int) -> str:
    chunks: list[bytes] = []
    total = 0
    try:
        while total <= 4096:
            chunk = os.read(descriptor, min(4097 - total, 4096))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
    except OSError:
        return ""
    finally:
        try:
            os.close(descriptor)
        except OSError:
            pass
    if total > 4096:
        return ""
    try:
        return b"".join(chunks).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return ""


def _run_provider(
    command: Sequence[str],
    *,
    workspace: Path,
    prompt: str,
    provider_name: str,
) -> _ProviderExecution:
    """Run one child with exact stdin/cwd and private-safe output replay."""

    stdout_sanitizer = AgentCLIStderrSanitizer()
    stderr_sanitizer = AgentCLIStderrSanitizer()
    trusted_read_fd = -1
    trusted_write_fd = -1
    popen_kwargs: dict[str, object] = {}
    child_env: dict[str, str] | None = None
    if provider_name.lower() == "grok" and _uses_packaged_grok_adapter(command):
        trusted_read_fd, trusted_write_fd = os.pipe()
        child_env = dict(os.environ)
        child_env[TRUSTED_FAILURE_RECEIPT_FD_ENV] = str(trusted_write_fd)
        popen_kwargs["env"] = child_env
        popen_kwargs["pass_fds"] = (trusted_write_fd,)
    try:
        process = subprocess.Popen(
            list(command),
            cwd=workspace,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            **popen_kwargs,
        )
    except OSError as exc:
        for descriptor in (trusted_read_fd, trusted_write_fd):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        diagnostic = stderr_sanitizer.feed(
            f"{provider_name} provider could not launch: {exc}\n"
        ) + stderr_sanitizer.finish()
        sys.stderr.write(diagnostic)
        sys.stderr.flush()
        return _ProviderExecution(
            AgentCLIProviderResult(
                None,
                launched=False,
                activity_state=AgentCLIActivityState.PRE_DISPATCH,
            )
        )
    if trusted_write_fd >= 0:
        os.close(trusted_write_fd)
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    stderr_tail = ""

    def replay_stdout() -> None:
        while True:
            chunk = process.stdout.read(8192)
            if not chunk:
                final = stdout_sanitizer.finish()
                if final:
                    sys.stdout.write(final)
                    sys.stdout.flush()
                return
            sanitized = stdout_sanitizer.feed(chunk)
            if sanitized:
                sys.stdout.write(sanitized)
                sys.stdout.flush()

    def replay_stderr() -> None:
        nonlocal stderr_tail
        while True:
            chunk = process.stderr.read(8192)
            if not chunk:
                final = stderr_sanitizer.finish()
                if final:
                    sys.stderr.write(final)
                    sys.stderr.flush()
                    stderr_tail = (stderr_tail + final)[-(256 * 1024) :]
                return
            sanitized = stderr_sanitizer.feed(chunk)
            if sanitized:
                sys.stderr.write(sanitized)
                sys.stderr.flush()
                stderr_tail = (stderr_tail + sanitized)[-(256 * 1024) :]

    stdout_thread = threading.Thread(
        target=replay_stdout,
        name=f"{provider_name}-stdout-replay",
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=replay_stderr,
        name=f"{provider_name}-stderr-replay",
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    try:
        process.stdin.write(prompt)
        process.stdin.flush()
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
    returncode = int(process.wait())
    stdout_thread.join()
    stderr_thread.join()
    trusted_receipt = (
        _read_private_failure_receipt(trusted_read_fd)
        if trusted_read_fd >= 0
        else ""
    )
    return _ProviderExecution(
        AgentCLIProviderResult(
            returncode,
            stderr=stderr_tail,
            launched=True,
            activity_state=AgentCLIActivityState.UNKNOWN,
        ),
        trusted_receipt,
    )


def _write_route_receipt(path: Path, record: str) -> None:
    encoded = record.encode("utf-8")
    if len(encoded) > 16 * 1024:
        raise LLMRouterError("provider route receipt exceeds body-free bound")
    destination = Path(os.path.abspath(path.expanduser()))
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            descriptor = -1
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run llm_router's fixed Grok -> Codex agent CLI route."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--primary-provider", required=True)
    parser.add_argument("--fallback-provider", required=True)
    parser.add_argument("--primary-command-json", required=True)
    parser.add_argument("--fallback-command-json", required=True)
    parser.add_argument(
        "--fallback-policy",
        choices=(AGENT_ROUTE_POLICY,),
        default=AGENT_ROUTE_POLICY,
    )
    parser.add_argument(
        "--primary-unavailable-kind",
        choices=(
            AgentCLIProviderFailureKind.AUTHENTICATION_FAILURE.value,
            AgentCLIProviderFailureKind.LAUNCH_FAILURE.value,
        ),
        default="",
    )
    parser.add_argument("--probe-route-readiness", action="store_true")
    parser.add_argument("--probe-grok-bin", default="")
    parser.add_argument("--probe-codex-bin", default="")
    parser.add_argument("--probe-grok-model", default="grok-4.5")
    parser.add_argument(
        "--probe-codex-model", default="gpt-5.6-terra"
    )
    parser.add_argument(
        "--probe-codex-reasoning-effort", default="high"
    )
    parser.add_argument("--route-receipt-path", type=Path)
    parser.add_argument("--route-task-id", default="")
    parser.add_argument("--route-attempt", type=int)
    parser.add_argument("--route-stage", default="")
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.probe_route_readiness and args.primary_unavailable_kind:
        print(
            "dynamic route readiness cannot be combined with a static "
            "primary-unavailable condition",
            file=sys.stderr,
        )
        return 2

    workspace = args.workspace.expanduser().resolve()
    if not workspace.is_dir():
        print(f"workspace is not a directory: {workspace}", file=sys.stderr)
        return 2
    try:
        primary_command = _command_from_json(
            args.primary_command_json,
            field_name="primary command",
            allow_empty=bool(args.primary_unavailable_kind),
        )
        fallback_command = _command_from_json(
            args.fallback_command_json,
            field_name="fallback command",
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    primary_unavailable_kind = str(args.primary_unavailable_kind or "")
    if args.probe_route_readiness:
        try:
            readiness = probe_grok_codex_agent_route_readiness(
                grok_bin=str(args.probe_grok_bin or ""),
                codex_bin=str(args.probe_codex_bin or ""),
                grok_model=str(args.probe_grok_model or "grok-4.5"),
                codex_model=str(
                    args.probe_codex_model or "gpt-5.6-terra"
                ),
                codex_reasoning_effort=str(
                    args.probe_codex_reasoning_effort or "high"
                ),
            )
        except Exception:
            print("agent route readiness probe failed terminally", file=sys.stderr)
            return 2
        if not readiness.codex_ready:
            print("Codex route fallback is not ready", file=sys.stderr)
            return 2
        if not readiness.grok_ready:
            failure_kind = getattr(readiness.failure_kind, "value", "")
            if failure_kind not in {
                AgentCLIProviderFailureKind.AUTHENTICATION_FAILURE.value,
                AgentCLIProviderFailureKind.LAUNCH_FAILURE.value,
            }:
                print(
                    "Grok route readiness probe failed terminally: "
                    f"{readiness.reason_code}",
                    file=sys.stderr,
                )
                return 2
            primary_unavailable_kind = failure_kind

    prompt = sys.stdin.read()
    primary_provider = safe_agent_cli_provider_label(
        str(args.primary_provider), default="primary"
    )
    fallback_provider = safe_agent_cli_provider_label(
        str(args.fallback_provider), default="fallback"
    )
    os.chdir(workspace)
    before = snapshot_agent_cli_workspace(workspace)
    if primary_unavailable_kind:
        primary_execution = _ProviderExecution(
            AgentCLIProviderResult(
                None,
                launched=False,
                activity_state=AgentCLIActivityState.PRE_DISPATCH,
            )
        )
        after = before
    else:
        primary_execution = _run_provider(
            primary_command,
            workspace=workspace,
            prompt=prompt,
            provider_name=primary_provider,
        )
        if primary_execution.result.returncode == 0:
            return 0
        after = snapshot_agent_cli_workspace(workspace)

    binding: dict[str, object] = {}
    if args.route_task_id:
        binding["task_id"] = args.route_task_id
    if args.route_attempt is not None:
        binding["attempt"] = args.route_attempt
    if args.route_stage:
        binding["stage"] = args.route_stage
    try:
        decision = route_agent_cli_failure(
            policy=args.fallback_policy,
            primary_provider=primary_provider,
            fallback_provider=fallback_provider,
            primary_result=primary_execution.result,
            workspace_before=before,
            workspace_after=after,
            trusted_failure_receipt=(
                primary_execution.trusted_failure_receipt
            ),
            primary_unavailable_kind=(primary_unavailable_kind or None),
            receipt_binding=binding,
        )
    except LLMRouterError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if not decision.should_fallback:
        print(
            f"{primary_provider} fallback suppressed: "
            f"{decision.classification.kind.value} "
            f"({decision.classification.reason_code}); "
            f"{decision.terminal_reason}",
            file=sys.stderr,
            flush=True,
        )
        return (
            127
            if primary_execution.result.returncode is None
            else primary_execution.result.returncode
        )

    route_record = serialize_agent_cli_route_record(decision)
    print(route_record, file=sys.stderr, flush=True)
    fallback_execution = _run_provider(
        fallback_command,
        workspace=workspace,
        prompt=prompt,
        provider_name=fallback_provider,
    )
    if args.route_receipt_path is not None:
        try:
            _write_route_receipt(args.route_receipt_path, route_record)
        except (OSError, LLMRouterError) as exc:
            print(
                f"provider route telemetry could not be persisted: {type(exc).__name__}",
                file=sys.stderr,
                flush=True,
            )
    return (
        127
        if fallback_execution.result.returncode is None
        else fallback_execution.result.returncode
    )


if __name__ == "__main__":
    raise SystemExit(main())
