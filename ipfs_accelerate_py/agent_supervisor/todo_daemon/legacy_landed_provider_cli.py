"""Exact no-fallback Grok/Codex adapters for legacy landed byte review.

The adapter sends ``LegacyLeafReviewRequest.canonical_prompt`` verbatim from a
fresh empty working directory.  It converts supervisor-observed child receipt
metadata into ``LegacyProviderObservation``; model output cannot author its
own effective-provider identity or execution receipt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import selectors
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from ..proof.formal_verification_contracts import content_identity
from .legacy_landed_review import (
    LEGACY_LANDED_LEAF_DECISION_SCHEMA,
    MAX_LEAF_TOKENS,
    LegacyLandedReviewPolicy,
    LegacyLeafReviewRequest,
    LegacyProviderCapacitySignal,
    LegacyProviderObservation,
    LegacyProviderPolicy,
)
from .llm import (
    LLM_USAGE_MODE_ENFORCE,
    LlmChildResultEnvelope,
    LlmRouterInvocation,
)

LEGACY_LANDED_CLI_EXECUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-cli-execution@2"
)
DEFAULT_LEGACY_LANDED_PROVIDER_TIMEOUT_SECONDS: Final = 300
DEFAULT_LEGACY_LANDED_MAX_NEW_TOKENS: Final = 1_024
LEGACY_LANDED_NATIVE_STRUCTURED_EXECUTION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-native-structured-execution@2"
)
_MAX_NATIVE_CLI_CAPTURE_BYTES: Final = 1024 * 1024
_MAX_NATIVE_RESPONSE_BYTES: Final = 64 * 1024
_MAX_NATIVE_FAILURE_CLASSIFICATION_BYTES: Final = 32 * 1024
_CODEX_USAGE_LIMIT_PATTERN: Final = re.compile(
    r"\byou(?:'|\u2019)ve hit your usage limit\b",
    re.IGNORECASE,
)
_GROK_BALANCE_EXHAUSTED_MESSAGE: Final = (
    "API error (status 402 Payment Required): Grok Build usage balance exhausted"
)
GROK_BUILD_BALANCE_EXHAUSTED_MARKER: Final = "grok_build_balance_exhausted"
_MAX_GROK_STREAM_EVENT_BYTES: Final = 64 * 1024
_NATIVE_CLI_SUBREAPER_PATH: Final = Path(__file__).with_name("native_cli_subreaper.py")


class NativeGrokQuotaExhaustionSignal(RuntimeError):
    """Fixed, secret-free signal from an exact Grok transport event.

    The signal is deliberately transport-specific.  Production policy may
    translate it into its typed quota exception, while model text and generic
    provider failures remain ordinary ``RuntimeError`` instances.
    """

    reason_code: Final = GROK_BUILD_BALANCE_EXHAUSTED_MARKER

    def __init__(self) -> None:
        super().__init__(self.reason_code)


LegacyCLIInvoker = Callable[
    [str, LlmRouterInvocation],
    tuple[str, LlmChildResultEnvelope | None],
]
NativeStructuredResponseValidator = Callable[
    [str, Mapping[str, Any]],
    Mapping[str, Any],
]


def _canonical_json(value: Mapping[str, Any]) -> str:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _leaf_decision_json_schema(
    request: LegacyLeafReviewRequest,
) -> dict[str, Any]:
    """Return the request-bound schema passed outside the canonical prompt."""

    manifest_id = request.payload.get("manifest_id")
    leaf = request.payload.get("leaf")
    leaf_id = leaf.get("leaf_id") if isinstance(leaf, Mapping) else None
    if not isinstance(manifest_id, str) or not manifest_id:
        raise RuntimeError("legacy native schema manifest binding is missing")
    if not isinstance(leaf_id, str) or not leaf_id:
        raise RuntimeError("legacy native schema leaf binding is missing")
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "schema": {
                "type": "string",
                "enum": [LEGACY_LANDED_LEAF_DECISION_SCHEMA],
            },
            "decision": {
                "type": "string",
                "enum": ["approve", "reject"],
            },
            "manifest_id": {"type": "string", "enum": [manifest_id]},
            "leaf_id": {"type": "string", "enum": [leaf_id]},
            "findings": {
                "type": "array",
                "items": {"type": "string"},
                # Keep the native schema in the flat common subset accepted
                # by both CLIs. Rejection remains an admissible fail-closed
                # decision, but native responses cannot attach detail.
                "maxItems": 0,
            },
        },
        "required": [
            "schema",
            "decision",
            "manifest_id",
            "leaf_id",
            "findings",
        ],
    }


def _last_json_object(value: str) -> dict[str, Any]:
    """Read one complete JSON object, allowing bounded CLI status lines."""

    raw = str(value or "").strip()
    candidates = [
        raw,
        *reversed([line.strip() for line in raw.splitlines() if line.strip()]),
    ]
    for candidate in candidates:
        if not candidate.startswith("{"):
            continue
        try:
            return _strict_json_object(candidate)
        except RuntimeError:
            continue
    raise RuntimeError("legacy native provider did not return a JSON object")


def _bounded_failure_sample(value: bytearray) -> bytes:
    """Return a bounded head/tail sample without retaining an unbounded copy."""

    if len(value) <= _MAX_NATIVE_FAILURE_CLASSIFICATION_BYTES:
        return bytes(value)
    head_bytes = _MAX_NATIVE_FAILURE_CLASSIFICATION_BYTES // 2
    tail_bytes = _MAX_NATIVE_FAILURE_CLASSIFICATION_BYTES - head_bytes - 1
    return bytes(value[:head_bytes]) + b"\n" + bytes(value[-tail_bytes:])


def _grok_stream_failure_kind(payload: Mapping[str, Any]) -> str:
    """Classify only CLI-owned structured failure event shapes."""

    if payload.get("type") == "error":
        if str(payload.get("message") or "").strip() == _GROK_BALANCE_EXHAUSTED_MESSAGE:
            return "verified_quota"
        return "other_failure"
    if payload.get("method") not in {
        "_x.ai/session/update",
        "session/update",
    }:
        return ""
    params = payload.get("params")
    update = params.get("update") if isinstance(params, Mapping) else None
    if not isinstance(update, Mapping):
        return ""
    if update.get("sessionUpdate") != "retry_state" or update.get("type") != "failed":
        return ""
    if (
        str(update.get("error_type") or "").strip().casefold() == "api"
        and str(update.get("message") or "").strip() == _GROK_BALANCE_EXHAUSTED_MESSAGE
    ):
        return "verified_quota"
    return "other_failure"


def _stdout_is_exact_grok_quota_failure(value: bytearray) -> bool:
    """Require a strict JSONL quota event with no conflicting failure.

    Grok is invoked with ``streaming-json`` output.  Therefore a non-empty
    non-JSON stdout line is a protocol failure, not evidence that authorizes a
    provider switch.  Stderr is intentionally never considered.
    """

    try:
        output = bytes(value).decode("utf-8", errors="strict")
    except UnicodeDecodeError:
        return False
    verified_quota = False
    for raw_line in output.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line.encode("utf-8")) > _MAX_GROK_STREAM_EVENT_BYTES:
            return False
        try:
            payload = _strict_json_object(line)
        except RuntimeError:
            return False
        failure_kind = _grok_stream_failure_kind(payload)
        if failure_kind == "verified_quota":
            verified_quota = True
        elif failure_kind == "other_failure":
            return False
    return verified_quota


def _native_cli_failure(
    command: Sequence[str],
    *,
    return_code: int,
    stdout: bytearray,
    stderr: bytearray,
) -> RuntimeError:
    """Classify exact capacity failures without exposing diagnostics."""

    executable = Path(str(command[0] or "")).name.casefold() if command else ""
    if (
        executable == "grok"
        and return_code == 1
        and _stdout_is_exact_grok_quota_failure(stdout)
    ):
        return NativeGrokQuotaExhaustionSignal()
    if executable == "codex":
        # ``codex exec --json`` versions have emitted the account-limit event
        # on either JSONL stdout or diagnostic stderr. Inspect bounded samples
        # of both, but return only a fixed typed token to the caller.
        for captured in (stdout, stderr):
            diagnostic = _bounded_failure_sample(captured).decode(
                "utf-8", errors="replace"
            )
            if _CODEX_USAGE_LIMIT_PATTERN.search(diagnostic):
                return LegacyProviderCapacitySignal()
    # Surface a small allowlisted set of secret-free transport statuses so
    # operator logs and capacity classifiers can distinguish max-turns cancels
    # from generic provider failures without dumping model transcripts.
    stderr_text = _bounded_failure_sample(stderr).decode("utf-8", errors="replace")
    stdout_text = _bounded_failure_sample(stdout).decode("utf-8", errors="replace")
    combined = f"{stderr_text}\n{stdout_text}".casefold()
    if "max turns reached" in combined:
        return RuntimeError("legacy native provider command failed: max turns reached")
    if "structuredoutputerror" in combined or "did not produce structured output" in combined:
        return RuntimeError(
            "legacy native provider command failed: structured output missing"
        )
    if "timed out" in combined or "timeout" in combined:
        return RuntimeError("legacy native provider command failed: timed out")
    return RuntimeError("legacy native provider command failed")


def _run_native_cli_process(
    command: Sequence[str],
    *,
    cwd: Path,
    timeout_seconds: int,
    stdin_text: str | None = None,
) -> tuple[str, str]:
    """Run one exact argv with bounded capture and subreaper confinement."""

    if (
        sys.platform != "linux"
        or not Path("/proc/self/task").is_dir()
        or not _NATIVE_CLI_SUBREAPER_PATH.is_file()
    ):
        # Native provider execution must never be attempted when the
        # fork/setsid confinement boundary cannot be established.
        raise RuntimeError("legacy native provider confinement unavailable")
    subreaper_command = [
        sys.executable,
        "-I",
        str(_NATIVE_CLI_SUBREAPER_PATH.resolve(strict=True)),
        "--",
        *[str(value) for value in command],
    ]

    # Bind every observed descendant PID to its Linux start time so cleanup
    # cannot accidentally signal a PID that was recycled during a long model
    # call. Tracking process groups also catches children which call setsid(2)
    # and escape the direct CLI's initial session.
    observed_members: dict[int, tuple[int, int]] = {}

    def process_identity(process_id: int) -> tuple[int, int] | None:
        try:
            raw = Path(f"/proc/{process_id}/stat").read_text(encoding="ascii")
            fields = raw[raw.rfind(")") + 2 :].split()
            return int(fields[2]), int(fields[19])  # pgrp, starttime
        except (IndexError, OSError, ValueError):
            return None

    def observe_family(root_process_id: int) -> None:
        pending = [root_process_id]
        visited: set[int] = set()
        while pending and len(visited) < 4_096:
            process_id = pending.pop()
            if process_id in visited:
                continue
            visited.add(process_id)
            identity = process_identity(process_id)
            if identity is None:
                continue
            observed_members[process_id] = identity
            try:
                task_paths = tuple(
                    Path(f"/proc/{process_id}/task").glob("*/children")
                )
            except OSError:
                task_paths = ()
            for children_path in task_paths:
                try:
                    children = children_path.read_text(encoding="ascii").split()
                except OSError:
                    continue
                for child in children:
                    try:
                        pending.append(int(child))
                    except ValueError:
                        continue

    def live_observed_groups() -> set[int]:
        groups: set[int] = set()
        for process_id, (recorded_group, recorded_start_time) in tuple(
            observed_members.items()
        ):
            current = process_identity(process_id)
            if current is not None and current[1] == recorded_start_time:
                groups.add(current[0] or recorded_group)
        return groups

    def process_group_exists(process_group_id: int) -> bool:
        try:
            os.killpg(process_group_id, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def terminate_family(process: subprocess.Popen[bytes]) -> None:
        # ``start_new_session`` makes the child's PID its process-group ID.
        # Signal that stable ID even when the direct CLI already exited: an
        # inherited stdout descriptor must not let a detached descendant
        # survive a timeout/non-zero/overflow result.
        process_group_id = int(process.pid)
        observe_family(process_group_id)
        process_groups = live_observed_groups() | {process_group_id}
        for group_id in process_groups:
            try:
                os.killpg(group_id, signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and any(
            process_group_exists(group_id) for group_id in process_groups
        ):
            try:
                process.wait(timeout=0.02)
            except subprocess.TimeoutExpired:
                pass
            time.sleep(0.01)
        remaining_groups = live_observed_groups() | {
            group_id
            for group_id in process_groups
            if process_group_exists(group_id)
        }
        for group_id in remaining_groups:
            try:
                os.killpg(group_id, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass

    stdin_handle = tempfile.TemporaryFile(mode="w+b")
    if stdin_text is not None:
        stdin_handle.write(stdin_text.encode("utf-8", errors="strict"))
        stdin_handle.seek(0)
    process: subprocess.Popen[bytes] | None = None
    selector = selectors.DefaultSelector()
    captures = {"stdout": bytearray(), "stderr": bytearray()}
    try:
        process = subprocess.Popen(
            subreaper_command,
            cwd=str(cwd),
            stdin=stdin_handle if stdin_text is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        assert process.stdout is not None
        assert process.stderr is not None
        observe_family(process.pid)
        for label, stream in (
            ("stdout", process.stdout),
            ("stderr", process.stderr),
        ):
            os.set_blocking(stream.fileno(), False)
            selector.register(stream, selectors.EVENT_READ, data=label)
        deadline = time.monotonic() + int(timeout_seconds)
        next_family_observation = time.monotonic()
        total_captured = 0
        while selector.get_map():
            now = time.monotonic()
            if now >= next_family_observation:
                observe_family(process.pid)
                next_family_observation = now + 0.25
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                terminate_family(process)
                raise RuntimeError("legacy native provider timed out")
            events = selector.select(timeout=min(remaining, 0.1))
            for key, _mask in events:
                try:
                    chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                except BlockingIOError:
                    continue
                if not chunk:
                    selector.unregister(key.fileobj)
                    key.fileobj.close()
                    continue
                total_captured += len(chunk)
                if total_captured > _MAX_NATIVE_CLI_CAPTURE_BYTES:
                    terminate_family(process)
                    raise RuntimeError(
                        "legacy native provider output exceeds capture bound"
                    )
                captures[str(key.data)].extend(chunk)
        try:
            return_code = process.wait(timeout=max(0.01, deadline - time.monotonic()))
        except subprocess.TimeoutExpired as exc:
            terminate_family(process)
            raise RuntimeError("legacy native provider timed out") from exc
        if return_code != 0:
            terminate_family(process)
            raise _native_cli_failure(
                command,
                return_code=return_code,
                stdout=captures["stdout"],
                stderr=captures["stderr"],
            )
        # A successful direct child may still have daemonized a descendant
        # after closing its pipes. The structured boundary permits no such
        # unobserved execution, so clean it before accepting the result.
        live_descendant = any(
            process_id != process.pid
            and process_identity(process_id) == identity
            for process_id, identity in tuple(observed_members.items())
        )
        if process_group_exists(process.pid) or live_descendant:
            terminate_family(process)
        try:
            stdout = bytes(captures["stdout"]).decode("utf-8", errors="strict")
            stderr = bytes(captures["stderr"]).decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("legacy native provider output is not UTF-8") from exc
        return stdout, stderr
    except BaseException:
        if process is not None:
            terminate_family(process)
        raise
    finally:
        selector.close()
        stdin_handle.close()
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if stream is not None and not stream.closed:
                    stream.close()


def _write_private_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")
    path.chmod(0o600)


def _read_bounded_regular_utf8(path: Path, *, max_bytes: int) -> str:
    """Read one no-follow regular file without allocating beyond its bound."""

    if (
        isinstance(max_bytes, bool)
        or not isinstance(max_bytes, int)
        or max_bytes < 1
    ):
        raise RuntimeError("legacy native response byte bound is invalid")
    descriptor = os.open(
        path,
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise RuntimeError("legacy native response is not a private regular file")
        if info.st_size > max_bytes:
            raise RuntimeError("legacy native structured response exceeds byte bound")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining > 0:
            chunk = os.read(descriptor, min(64 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        response_bytes = b"".join(chunks)
        if len(response_bytes) > max_bytes:
            raise RuntimeError("legacy native structured response exceeds byte bound")
        # Reject a concurrent replace/growth race even if the bounded prefix
        # happened to parse as JSON.
        after = os.fstat(descriptor)
        if (
            after.st_dev != info.st_dev
            or after.st_ino != info.st_ino
            or after.st_size != len(response_bytes)
        ):
            raise RuntimeError("legacy native structured response changed while read")
        return response_bytes.decode("utf-8", errors="strict").strip()
    except UnicodeDecodeError as exc:
        raise RuntimeError("legacy Codex CLI structured result is unreadable") from exc
    finally:
        os.close(descriptor)


def _grok_native_structured_output(
    prompt: str,
    invocation: LlmRouterInvocation,
    response_schema: Mapping[str, Any],
) -> tuple[str, str]:
    binary = shutil.which("grok")
    if not binary:
        raise RuntimeError("legacy Grok CLI is unavailable")
    cwd = invocation.repo_root.resolve(strict=True)
    prompt_path = cwd / "canonical-prompt.json"
    # Grok's ``--prompt-file`` accepts an ACP content envelope, not a raw text
    # file.  Keep the canonical supervisor prompt as the one verbatim text
    # block; the private wrapper is transport metadata and is never part of
    # the model-visible prompt text.
    prompt_file_payload = _canonical_json(
        {
            "type": "acp",
            "content": [{"type": "text", "text": prompt}],
        }
    )
    _write_private_text(prompt_path, prompt_file_payload)
    if prompt_path.read_bytes() != prompt_file_payload.encode(
        "utf-8", errors="strict"
    ):
        raise RuntimeError("legacy canonical request changed in prompt file")
    # Production implement prompts carry a full typed packet.  A single turn is
    # often cancelled before constrained structured output is finalized
    # (``structuredOutputError: model did not produce structured output`` with
    # ``Error: max turns reached``).  Allow a small fixed turn budget while
    # still forbidding tools/subagents/web so the side-effect boundary holds.
    command = [
        binary,
        "--model",
        invocation.model_name,
        "--json-schema",
        _canonical_json(response_schema),
        "--output-format",
        "json",
        "--no-plan",
        "--no-subagents",
        "--disable-web-search",
        "--no-memory",
        "--verbatim",
        "--max-turns",
        "8",
        "--permission-mode",
        "dontAsk",
        "--tools",
        "",
        "--prompt-file",
        str(prompt_path),
    ]
    stdout, _stderr = _run_native_cli_process(
        command,
        cwd=cwd,
        timeout_seconds=invocation.timeout_seconds,
    )
    payload = _last_json_object(stdout)
    if str(payload.get("type") or "").casefold() == "error":
        raise RuntimeError("legacy Grok CLI returned an error result")
    # Prefer the CLI's structuredOutput envelope when present: current Grok
    # builds emit the constrained schema object there while also stuffing a
    # human-readable JSON string under ``text``.
    structured = payload.get("structuredOutput")
    structured_error = str(payload.get("structuredOutputError") or "").strip()
    if isinstance(structured, Mapping) and structured:
        response_text = _canonical_json(structured)
    elif "text" in payload:
        response_value = payload.get("text")
        if isinstance(response_value, Mapping):
            response_text = _canonical_json(response_value)
        elif isinstance(response_value, str) and response_value.strip():
            response_text = response_value.strip()
        else:
            detail = structured_error or "structured result is missing"
            raise RuntimeError(f"legacy Grok CLI structured result failed: {detail}")
    else:
        # Current Grok releases may emit the schema object directly, while
        # older releases wrap it in ``text``. The caller's strict validator
        # decides whether the direct object is the requested response shape.
        if structured_error and not any(
            key in payload for key in ("packet_id", "proposal", "decision")
        ):
            raise RuntimeError(
                f"legacy Grok CLI structured result failed: {structured_error}"
            )
        response_text = _canonical_json(payload)
    endpoint_value = payload.get("requestId")
    endpoint_receipt_id = ""
    if isinstance(endpoint_value, str) and endpoint_value:
        endpoint_receipt_id = content_identity(
            {
                "provider": invocation.provider,
                "model": invocation.model_name,
                "endpoint_request_sha256": hashlib.sha256(
                    endpoint_value.encode("utf-8")
                ).hexdigest(),
            }
        )
    return response_text, endpoint_receipt_id


def _codex_native_structured_output(
    prompt: str,
    invocation: LlmRouterInvocation,
    response_schema: Mapping[str, Any],
    *,
    max_response_bytes: int,
    reasoning_effort: str,
) -> tuple[str, str]:
    binary = shutil.which("codex")
    if not binary:
        raise RuntimeError("legacy Codex CLI is unavailable")
    cwd = invocation.repo_root.resolve(strict=True)
    reasoning_effort = str(reasoning_effort or "").strip()
    if reasoning_effort != "medium":
        raise RuntimeError(
            "legacy Codex independent review requires medium reasoning"
        )
    schema_path = cwd / "response-schema.json"
    response_path = cwd / "last-message.json"
    _write_private_text(schema_path, _canonical_json(response_schema))
    _write_private_text(response_path, "")
    command = [
        binary,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--model",
        invocation.model_name,
        "-c",
        f'model_reasoning_effort="{reasoning_effort}"',
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(response_path),
        "--json",
        "-",
    ]
    _stdout, _stderr = _run_native_cli_process(
        command,
        cwd=cwd,
        timeout_seconds=invocation.timeout_seconds,
        stdin_text=prompt,
    )
    try:
        response_text = _read_bounded_regular_utf8(
            response_path,
            max_bytes=max_response_bytes,
        )
    except OSError as exc:
        raise RuntimeError("legacy Codex CLI structured result is unreadable") from exc
    if not response_text:
        raise RuntimeError("legacy Codex CLI structured result is missing")
    return response_text, ""


def _validate_native_response(
    response_text: str,
    response_schema: Mapping[str, Any],
) -> dict[str, Any]:
    if len(response_text.encode("utf-8")) > _MAX_NATIVE_RESPONSE_BYTES:
        raise RuntimeError("legacy native structured response exceeds byte bound")
    response = _strict_json_object(response_text)
    properties = response_schema.get("properties")
    if not isinstance(properties, Mapping):
        raise RuntimeError("legacy native response schema is invalid")
    manifest_values = properties.get("manifest_id")
    leaf_values = properties.get("leaf_id")
    expected_manifest = (
        manifest_values.get("enum")
        if isinstance(manifest_values, Mapping)
        else None
    )
    expected_leaf = (
        leaf_values.get("enum") if isinstance(leaf_values, Mapping) else None
    )
    if (
        set(response)
        != {"schema", "decision", "manifest_id", "leaf_id", "findings"}
        or response.get("schema") != LEGACY_LANDED_LEAF_DECISION_SCHEMA
        or response.get("decision") not in {"approve", "reject"}
        or expected_manifest != [response.get("manifest_id")]
        or expected_leaf != [response.get("leaf_id")]
        or not isinstance(response.get("findings"), list)
        or len(response["findings"]) > 64
        or not all(isinstance(item, str) for item in response["findings"])
        or response["findings"] != []
    ):
        raise RuntimeError("legacy native structured response violates its schema")
    return response


def _invoke_native_structured_cli(
    prompt: str,
    invocation: LlmRouterInvocation,
    response_schema: Mapping[str, Any],
    *,
    response_validator: NativeStructuredResponseValidator | None = None,
    execution_schema: str = LEGACY_LANDED_NATIVE_STRUCTURED_EXECUTION_SCHEMA,
    max_response_bytes: int = _MAX_NATIVE_RESPONSE_BYTES,
    codex_reasoning_effort: str = "",
) -> tuple[str, LlmChildResultEnvelope]:
    if (
        isinstance(max_response_bytes, bool)
        or not isinstance(max_response_bytes, int)
        or max_response_bytes < 1
        or max_response_bytes > _MAX_NATIVE_CLI_CAPTURE_BYTES
    ):
        raise RuntimeError("legacy native response byte bound is invalid")
    expected_provider = str(invocation.provider or "")
    if expected_provider == "grok_cli":
        response_text, endpoint_receipt_id = _grok_native_structured_output(
            prompt,
            invocation,
            response_schema,
        )
    elif expected_provider == "codex_cli":
        response_text, endpoint_receipt_id = _codex_native_structured_output(
            prompt,
            invocation,
            response_schema,
            max_response_bytes=max_response_bytes,
            reasoning_effort=codex_reasoning_effort,
        )
    else:
        raise RuntimeError("legacy native provider is not policy-admissible")
    if len(response_text.encode("utf-8")) > max_response_bytes:
        raise RuntimeError("legacy native structured response exceeds byte bound")
    validator = response_validator or _validate_native_response
    response = dict(validator(response_text, response_schema))
    response_text = _canonical_json(response)
    if len(response_text.encode("utf-8")) > max_response_bytes:
        raise RuntimeError("legacy native structured response exceeds byte bound")
    execution_body = {
        "schema": str(execution_schema),
        "request_id": invocation.request_id,
        "configured_provider": expected_provider,
        "configured_model": invocation.model_name,
        "model_reasoning_effort": codex_reasoning_effort,
        "effective_provider": expected_provider,
        "effective_model": invocation.model_name,
        "canonical_prompt_sha256": hashlib.sha256(
            prompt.encode("utf-8", errors="strict")
        ).hexdigest(),
        "output_schema_id": content_identity(response_schema),
        "response_id": content_identity(response),
        "exit_code": 0,
    }
    execution_result_id = content_identity(execution_body)
    supervisor_receipt_id = content_identity(
        {
            "schema": "ipfs_accelerate_py/agent-supervisor/native-cli-receipt@1",
            "execution_result_id": execution_result_id,
            "provider": expected_provider,
            "model": invocation.model_name,
        }
    )
    return response_text, LlmChildResultEnvelope(
        usage_mode=invocation.usage_mode,
        request_id=invocation.request_id,
        attempt=invocation.attempt,
        idempotency_key=invocation.idempotency_key,
        status="ok",
        supervisor_receipt_id=supervisor_receipt_id,
        endpoint_receipt_id=endpoint_receipt_id,
        execution_result_id=execution_result_id,
        effective_provider=expected_provider,
        text_chars=len(response_text),
        exit_code=0,
    )


def _strict_json_object(value: str) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in items:
            if key in result:
                raise RuntimeError("legacy provider response contains duplicate fields")
            result[key] = item
        return result

    try:
        parsed = json.loads(
            value,
            object_pairs_hook=pairs,
            parse_constant=lambda item: (_ for _ in ()).throw(
                ValueError(f"non-finite JSON value: {item}")
            ),
        )
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError("legacy provider response is not strict JSON") from exc
    if not isinstance(parsed, dict):
        raise RuntimeError("legacy provider response must contain an object")
    return parsed


@dataclass(frozen=True, slots=True)
class BoundLegacyLandedCLIProvider:
    """One operator-policy-bound effective CLI provider."""

    provider_policy: LegacyProviderPolicy
    timeout_seconds: int = DEFAULT_LEGACY_LANDED_PROVIDER_TIMEOUT_SECONDS
    max_new_tokens: int = DEFAULT_LEGACY_LANDED_MAX_NEW_TOKENS
    invoker: LegacyCLIInvoker | None = None

    def __post_init__(self) -> None:
        if self.provider_policy.role not in {"grok_audit", "codex_audit"}:
            raise ValueError("legacy CLI adapter role is invalid")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, int)
            or not 1 <= self.timeout_seconds <= 300
        ):
            raise ValueError("legacy CLI timeout must be in [1, 300]")
        if (
            isinstance(self.max_new_tokens, bool)
            or not isinstance(self.max_new_tokens, int)
            or not 1 <= self.max_new_tokens <= 4_096
        ):
            raise ValueError("legacy CLI response token bound is invalid")

    def _invoke(
        self,
        prompt: str,
        invocation: LlmRouterInvocation,
        response_schema: Mapping[str, Any],
    ) -> tuple[str, LlmChildResultEnvelope | None]:
        if self.invoker is not None:
            return self.invoker(prompt, invocation)
        return _invoke_native_structured_cli(
            prompt,
            invocation,
            response_schema,
            codex_reasoning_effort=self.provider_policy.reasoning_effort,
        )

    def __call__(
        self, request: LegacyLeafReviewRequest
    ) -> LegacyProviderObservation:
        if not isinstance(request, LegacyLeafReviewRequest):
            raise TypeError("legacy CLI provider requires LegacyLeafReviewRequest")
        expected = self.provider_policy
        if (
            request.role != expected.role
            or request.provider != expected.provider
            or request.model != expected.model
            or request.reasoning_effort != expected.reasoning_effort
        ):
            raise RuntimeError("legacy CLI request differs from operator policy")
        if request.token_upper_bound > MAX_LEAF_TOKENS:
            raise RuntimeError("legacy CLI full request exceeds 4096 tokens")
        try:
            prompt = request.canonical_prompt.decode("ascii", errors="strict")
        except UnicodeDecodeError as exc:
            raise RuntimeError("legacy canonical request must be ASCII DAG-JSON") from exc
        if prompt.encode("ascii") != request.canonical_prompt:
            raise RuntimeError("legacy canonical request changed before invocation")
        response_schema = _leaf_decision_json_schema(request)

        with tempfile.TemporaryDirectory(
            prefix=f"ipfs-accelerate-{expected.role}-"
        ) as temporary_cwd:
            invocation = LlmRouterInvocation(
                repo_root=Path(temporary_cwd).resolve(),
                model_name=expected.model,
                provider=expected.provider,
                allow_local_fallback=False,
                allow_cross_provider_fallback=False,
                timeout_seconds=self.timeout_seconds,
                max_new_tokens=self.max_new_tokens,
                max_prompt_chars=len(prompt),
                temperature=0.0,
                python_executable=sys.executable,
                timeout_grace_seconds=0,
                trace=True,
                reject_effective_provider_name=None,
                required_effective_providers=(expected.provider,),
                usage_mode=LLM_USAGE_MODE_ENFORCE,
                request_id=request.request_id,
                attempt=1,
                idempotency_key=request.request_id,
                side_effect_boundary="review_only",
                write_result_envelope=True,
            )
            output, child = self._invoke(prompt, invocation, response_schema)

        if child is None:
            raise RuntimeError("legacy provider child receipt is missing")
        receipt = child.to_dict()
        exit_code = receipt.get("exit_code")
        if (
            receipt.get("status") != "ok"
            or isinstance(exit_code, bool)
            or not isinstance(exit_code, int)
            or exit_code != 0
            or receipt.get("request_id") != request.request_id
            or receipt.get("idempotency_key") != request.request_id
            or receipt.get("effective_provider") != expected.provider
        ):
            raise RuntimeError("legacy provider child receipt is not exactly bound")
        response = _strict_json_object(output)
        observation_body = {
            "schema": LEGACY_LANDED_CLI_EXECUTION_SCHEMA,
            "request_id": request.request_id,
            "role": expected.role,
            "configured_provider": expected.provider,
            "configured_model": expected.model,
            "effective_provider": receipt["effective_provider"],
            "effective_model": expected.model,
            "child_result_schema": receipt["schema"],
            "child_result_status": receipt["status"],
            "child_exit_code": exit_code,
            "supervisor_receipt_id": str(receipt.get("supervisor_receipt_id") or ""),
            "endpoint_receipt_id": str(receipt.get("endpoint_receipt_id") or ""),
            "execution_result_id": str(receipt.get("execution_result_id") or ""),
            "response_id": content_identity(response),
            "native_output_schema_id": content_identity(response_schema),
            "native_structured_output_enforced": self.invoker is None,
            "full_request_token_upper_bound": request.token_upper_bound,
            "fallback_used": False,
            "repository_checkout_used_as_working_directory": False,
            "model_output_authored_execution_receipt": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
        return LegacyProviderObservation(
            observation_id=content_identity(observation_body),
            requested_provider=expected.provider,
            requested_model=expected.model,
            effective_provider=expected.provider,
            effective_model=expected.model,
            provider_chain=(expected.provider,),
            fallback_used=False,
            supervisor_observed=True,
            response=response,
            requested_reasoning_effort=expected.reasoning_effort,
            effective_reasoning_effort=expected.reasoning_effort,
        )


def build_legacy_landed_cli_provider_pair(
    policy: LegacyLandedReviewPolicy,
    *,
    invoker: LegacyCLIInvoker | None = None,
) -> tuple[BoundLegacyLandedCLIProvider, BoundLegacyLandedCLIProvider]:
    """Build distinct exact Grok and Codex audit callables from operator policy."""

    if not isinstance(policy, LegacyLandedReviewPolicy):
        raise TypeError("parsed legacy landed review policy is required")
    return (
        BoundLegacyLandedCLIProvider(policy.grok, invoker=invoker),
        BoundLegacyLandedCLIProvider(policy.codex, invoker=invoker),
    )


__all__ = [
    "DEFAULT_LEGACY_LANDED_MAX_NEW_TOKENS",
    "DEFAULT_LEGACY_LANDED_PROVIDER_TIMEOUT_SECONDS",
    "LEGACY_LANDED_CLI_EXECUTION_SCHEMA",
    "LEGACY_LANDED_NATIVE_STRUCTURED_EXECUTION_SCHEMA",
    "BoundLegacyLandedCLIProvider",
    "LegacyCLIInvoker",
    "build_legacy_landed_cli_provider_pair",
]
