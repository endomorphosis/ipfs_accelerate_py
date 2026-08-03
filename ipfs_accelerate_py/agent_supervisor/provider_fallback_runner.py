#!/usr/bin/env python3
"""Run one stdin-driven provider command with one ordered fallback.

The implementation daemon supplies commands as JSON argument vectors.  This
runner deliberately does not use a shell: both children receive the exact same
stdin prompt, have their output replayed to this process's stdout/stderr
streams, and run in the same resolved workspace.  The default compatibility
policy falls back after any primary failure.  Callers may instead opt into the
narrow ``grok_quota_exhausted`` policy, which preserves every non-quota Grok
failure and invokes Codex only after typed positive quota classification.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

ANY_FAILURE_POLICY = "any_failure"
GROK_QUOTA_EXHAUSTED_POLICY = "grok_quota_exhausted"


class GrokFailureKind(str, Enum):
    """Finite result of classifying one failed Grok provider process."""

    QUOTA_EXHAUSTED = "grok_quota_exhausted"
    AUTHENTICATION_FAILURE = "authentication_failure"
    LAUNCH_FAILURE = "launch_failure"
    TIMEOUT = "timeout"
    TRANSPORT_FAILURE = "transport_failure"
    MALFORMED_OUTPUT = "malformed_output"
    NONZERO_EXIT = "generic_nonzero_exit"


@dataclass(frozen=True)
class ProviderRunResult:
    """Captured provider outcome; ``None`` means the process did not launch."""

    returncode: int | None
    stdout: str = ""
    stderr: str = ""


@dataclass(frozen=True)
class GrokFailureClassification:
    """Typed fail-closed classification used by quota-only fallback policy."""

    kind: GrokFailureKind
    reason_code: str

    @property
    def confirms_quota_exhaustion(self) -> bool:
        return self.kind is GrokFailureKind.QUOTA_EXHAUSTED


_AUTH_FAILURE_PATTERN = re.compile(
    r"(?:\b(?:unauthenticated|unauthorized)\b|"
    r"\bauthentication\s+(?:failed|required)\b|"
    r"\b(?:invalid|missing|expired)\s+(?:xai\s+)?api[_ -]?key\b|"
    r"\b(?:login|required to log in|not logged in)\b|"
    r"\b(?:http|status(?:\s+code)?)\s*[:=]?\s*(?:401|403)\b)",
    re.IGNORECASE,
)
_TIMEOUT_PATTERN = re.compile(
    r"\b(?:timed?\s*out|timeout|deadline\s+exceeded)\b",
    re.IGNORECASE,
)
_TRANSPORT_FAILURE_PATTERN = re.compile(
    r"(?:\bconnection\s+(?:refused|reset|aborted|closed)\b|"
    r"\b(?:dns|tls|network|transport|socket)\s+(?:error|failure)\b|"
    r"\btemporary failure in name resolution\b|"
    r"\bno route to host\b)",
    re.IGNORECASE,
)
_PLAIN_QUOTA_PATTERNS = (
    re.compile(
        r"^\s*(?:error\s*:\s*)?(?:you(?:'|\u2019)?ve|you have)\s+hit\s+"
        r"your\s+(?:grok\s+|xai\s+)?usage\s+limit\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:error\s*:\s*)?(?:(?:grok|xai)(?:\s+api)?\s+)?"
        r"(?:account\s+|organization\s+)?(?:usage\s+)?quota\s+"
        r"(?:is\s+|has\s+been\s+)?(?:exhausted|exceeded|depleted)\.?\s*$",
        re.IGNORECASE,
    ),
    re.compile(
        r"^\s*(?:error\s*:\s*)?(?:grok|xai)(?:\s+api)?\b.*\b"
        r"(?:usage|credit)\s+(?:quota|balance|limit)\b.*\b"
        r"(?:exhausted|exceeded|depleted|reached)\b.*$",
        re.IGNORECASE,
    ),
)
_STRUCTURED_QUOTA_CODES = frozenset(
    {
        "billing_hard_limit_reached",
        "credit_balance_exhausted",
        "insufficient_quota",
        "quota_exhausted",
        "usage_limit_reached",
    }
)


def _structured_quota_code(output: str) -> str:
    """Return an exact provider quota code from a valid JSON error line."""

    def visit(value: object, *, inside_error: bool = False) -> str:
        if isinstance(value, dict):
            for raw_key, child in value.items():
                key = str(raw_key).strip().lower()
                nested_error = inside_error or key in {"error", "errors"}
                if nested_error and key in {"code", "reason", "type"}:
                    candidate = str(child).strip().lower().replace("-", "_")
                    if candidate in _STRUCTURED_QUOTA_CODES:
                        return candidate
                found = visit(child, inside_error=nested_error)
                if found:
                    return found
        elif isinstance(value, list):
            for child in value:
                found = visit(child, inside_error=inside_error)
                if found:
                    return found
        return ""

    for line in output.splitlines():
        candidate = line.strip()
        if not candidate.startswith(("{", "[")):
            continue
        try:
            payload = json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
        code = visit(payload)
        if code:
            return code
    return ""


def classify_grok_failure(result: ProviderRunResult) -> GrokFailureClassification:
    """Classify a failed Grok run without inferring quota from generic errors."""

    if result.returncode is None:
        return GrokFailureClassification(
            GrokFailureKind.LAUNCH_FAILURE,
            "grok_process_did_not_launch",
        )
    # Grok agent stdout may contain arbitrary task/tool output.  Only the
    # provider-owned diagnostic channel may positively authorize fallback.
    output = result.stderr
    if "\x00" in output or "\ufffd" in output:
        return GrokFailureClassification(
            GrokFailureKind.MALFORMED_OUTPUT,
            "grok_output_not_valid_text",
        )
    if _AUTH_FAILURE_PATTERN.search(output):
        return GrokFailureClassification(
            GrokFailureKind.AUTHENTICATION_FAILURE,
            "grok_authentication_failure",
        )
    if _TIMEOUT_PATTERN.search(output):
        return GrokFailureClassification(
            GrokFailureKind.TIMEOUT,
            "grok_timeout",
        )
    if _TRANSPORT_FAILURE_PATTERN.search(output):
        return GrokFailureClassification(
            GrokFailureKind.TRANSPORT_FAILURE,
            "grok_transport_failure",
        )
    structured_code = _structured_quota_code(output)
    if structured_code:
        return GrokFailureClassification(
            GrokFailureKind.QUOTA_EXHAUSTED,
            f"grok_provider_{structured_code}",
        )
    if any(
        pattern.fullmatch(line)
        for line in output.splitlines()
        for pattern in _PLAIN_QUOTA_PATTERNS
    ):
        return GrokFailureClassification(
            GrokFailureKind.QUOTA_EXHAUSTED,
            "grok_provider_plain_quota_exhausted",
        )
    return GrokFailureClassification(
        GrokFailureKind.NONZERO_EXIT,
        "grok_non_quota_failure",
    )


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
) -> ProviderRunResult:
    """Run and replay one provider, retaining a bounded stderr tail."""

    try:
        process = subprocess.Popen(
            list(command),
            cwd=workspace,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as exc:
        print(
            f"{provider_name} provider could not launch: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return ProviderRunResult(returncode=None)
    assert process.stdin is not None
    assert process.stderr is not None
    stderr_tail = ""

    def replay_stderr() -> None:
        nonlocal stderr_tail
        while True:
            chunk = process.stderr.read(8192)
            if not chunk:
                return
            sys.stderr.write(chunk)
            sys.stderr.flush()
            stderr_tail = (stderr_tail + chunk)[-(256 * 1024) :]

    stderr_thread = threading.Thread(
        target=replay_stderr,
        name=f"{provider_name}-stderr-replay",
        daemon=True,
    )
    stderr_thread.start()
    try:
        process.stdin.write(prompt)
        process.stdin.flush()
    except BrokenPipeError:
        pass
    finally:
        process.stdin.close()
    returncode = int(process.wait())
    stderr_thread.join()
    return ProviderRunResult(
        returncode=returncode,
        stderr=stderr_tail,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a primary implementation provider with one fallback."
    )
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--primary-provider", required=True)
    parser.add_argument("--fallback-provider", required=True)
    parser.add_argument("--primary-command-json", required=True)
    parser.add_argument("--fallback-command-json", required=True)
    parser.add_argument(
        "--fallback-policy",
        choices=(ANY_FAILURE_POLICY, GROK_QUOTA_EXHAUSTED_POLICY),
        default=ANY_FAILURE_POLICY,
    )
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
    primary_result = _run_provider(
        primary_command,
        workspace=workspace,
        prompt=prompt,
        provider_name=primary_provider,
    )
    if primary_result.returncode == 0:
        return 0

    if args.fallback_policy == GROK_QUOTA_EXHAUSTED_POLICY:
        if primary_provider.lower() != "grok":
            print(
                "grok_quota_exhausted fallback policy requires Grok as the "
                "primary provider",
                file=sys.stderr,
                flush=True,
            )
            return 2
        classification = classify_grok_failure(primary_result)
        if not classification.confirms_quota_exhaustion:
            print(
                f"{primary_provider} fallback suppressed by quota-only policy: "
                f"{classification.kind.value} "
                f"({classification.reason_code})",
                file=sys.stderr,
                flush=True,
            )
            return (
                127
                if primary_result.returncode is None
                else primary_result.returncode
            )
        print(
            f"{primary_provider} quota exhaustion confirmed "
            f"({classification.reason_code}); falling back to "
            f"{fallback_provider}",
            file=sys.stderr,
            flush=True,
        )
    elif primary_result.returncode is not None:
        print(
            f"{primary_provider} provider exited with "
            f"{primary_result.returncode}; falling back to {fallback_provider}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"falling back to {fallback_provider}",
            file=sys.stderr,
            flush=True,
        )

    fallback_result = _run_provider(
        fallback_command,
        workspace=workspace,
        prompt=prompt,
        provider_name=fallback_provider,
    )
    return (
        127
        if fallback_result.returncode is None
        else fallback_result.returncode
    )


if __name__ == "__main__":
    raise SystemExit(main())
