"""
OpenAI Codex CLI Integration (compatibility facade)

Uses current ``codex exec`` semantics rather than the obsolete
``openai api completions.create`` command. Operator command overrides are
argv-only (shell-free). Side-effecting exec runs disable generic cache/retry.
"""

from __future__ import annotations

import logging
import contextlib
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .base_cli_wrapper import (
    BaseCLIWrapper,
    parse_argv_override,
    resolve_command_override_from_env,
)
from ..common.llm_cache import LLMAPICache, get_llm_cache

logger = logging.getLogger(__name__)

# Env vars (first wins) for full argv override of the codex binary/prefix.
_CODEX_CMD_ENV = (
    "ipfs_accelerate_py_CODEX_CLI_CMD",
    "IPFS_ACCELERATE_PY_CODEX_CLI_CMD",
    "IPFS_DATASETS_PY_CODEX_CLI_CMD",
)


def _default_codex_path() -> str:
    return shutil.which("codex") or "codex"


def build_codex_exec_argv(
    *,
    base_argv: Sequence[str],
    model: str,
    prompt: str,
    last_message_path: str,
    sandbox: Optional[str] = None,
    skip_git_repo_check: bool = True,
    json_mode: bool = False,
) -> List[str]:
    """Build ``codex exec`` argv matching llm_router production semantics.

    The prompt is *not* placed on the argv; callers must pass it via stdin
    (``cmd.append("-")``). Dynamic values remain single argv entries.
    """
    _ = prompt  # documented for callers; supplied via stdin
    cmd: List[str] = list(base_argv)
    # If operator override is a bare binary, append the exec subcommand.
    # If override already includes "exec", do not duplicate it.
    lowered = [part.lower() for part in cmd]
    if "exec" not in lowered:
        cmd.append("exec")
    if skip_git_repo_check:
        cmd.append("--skip-git-repo-check")
    # Treat sandbox "auto"/empty as omit-flag (CLI chooses default).
    if sandbox and str(sandbox).strip().lower() not in {"", "auto"}:
        cmd.extend(["--sandbox", str(sandbox).strip()])
    if model:
        cmd.extend(["-m", str(model)])
    cmd.extend(["--output-last-message", last_message_path])
    if json_mode:
        cmd.append("--json")
    cmd.append("-")
    return cmd


def _open_codex_app_connection(socket_path, *, timeout_seconds, max_response_bytes):
    """Connect to an explicitly configured, existing owned daemon; never start one."""
    info = os.stat(socket_path, follow_symlinks=False)
    if not stat.S_ISSOCK(info.st_mode) or info.st_uid != os.geteuid():
        raise ValueError("daemon_socket_unavailable")
    from websockets.sync.client import unix_connect

    return unix_connect(
        str(socket_path), proxy=None, open_timeout=timeout_seconds,
        close_timeout=min(1.0, timeout_seconds), max_size=max_response_bytes,
        max_queue=4,
    )


def send_codex_thread_message(
    *, socket_path, thread_id, expected_turn_id, message, receipt_path,
    timeout_seconds=15.0, max_attempts=3, prior_delivery_state="not_attempted",
) -> Dict[str, Any]:
    """Bounded handoff to an existing thread, independent of the HTTP MCP bridge.

    The caller supplies an existing owned0700 receipt directory and the current
    target turn. One invocation makes at most one connection attempt. The receipt
    reserves that attempt before connecting and records UNKNOWN before steering.
    Accepted and ambiguous sends are never replayed, including after a restart.
    A caller whose earlier channel may have sent must pass ``unknown``; this
    function cannot infer non-delivery from a generic transport exception.

    The caller must retain this receipt namespace and lock entries for the
    operation and future retries. Do not clean or replace them as part of a
    recovery. Path identities are rechecked before persistence and dispatch;
    this cooperative check is not protection against an owner deleting the
    namespace during the send itself.

    Acknowledgement means enqueue only. It confers no task claim, execution,
    resource, completion, or legal-review authority. No model/process is started,
    and generation's existing no-cache/no-retry behavior is unchanged.
    """
    result = {
        "delivery_state": "refused", "reason": "invalid_request",
        "enqueue_acknowledged": False, "processing_confirmed": False,
        "completion_eligible": False, "execution_granted": False,
        "claim_granted": False, "retryable": False, "attempts": 0,
    }
    try:
        if not all(type(v) is str and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", v)
                   for v in (thread_id, expected_turn_id)):
            return result
        if type(message) is not str or not 0 < len(message.encode("utf-8")) <= 1000:
            return result
        if (type(timeout_seconds) not in (int, float) or not math.isfinite(timeout_seconds)
                or not 0 < timeout_seconds <= 60 or type(max_attempts) is not int
                or not 1 <= max_attempts <= 8
                or prior_delivery_state not in ("not_attempted", "not_sent", "unknown")):
            return result
        endpoint, receipt = Path(socket_path), Path(receipt_path)
        if (not endpoint.is_absolute() or ".." in endpoint.parts
                or "\0" in str(endpoint) or len(os.fsencode(endpoint)) > 107
                or not receipt.is_absolute() or ".." in receipt.parts
                or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\.json", receipt.name)):
            return result
        identity = {
            "schema": "codex-existing-thread-delivery/v1", "socket_path": str(endpoint),
            "thread_id": thread_id, "expected_turn_id": expected_turn_id,
            "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
            "max_attempts": max_attempts,
        }
        if len(json.dumps(identity).encode("utf-8")) > 3072:
            return result
    except (TypeError, ValueError, UnicodeError):
        return result

    try:
        import fcntl
    except ImportError:
        return {**result, "reason": "unix_handoff_unavailable"}

    deadline = time.monotonic() + timeout_seconds

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("handoff_deadline")
        return left

    try:
        with contextlib.ExitStack() as stack:
            directories = []
            def directory_identity(info):
                return (info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid)

            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            stack.callback(os.close, directory)
            for part in receipt.parent.parts[1:]:
                parent = directory
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                                    | os.O_CLOEXEC, dir_fd=directory)
                stack.callback(os.close, directory)
                directories.append((parent, part, directory,
                                    directory_identity(os.fstat(directory))))
            info = os.fstat(directory)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                return {**result, "reason": "receipt_directory_not_private"}
            lock = os.open(receipt.name + ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW
                           | os.O_CLOEXEC | os.O_NONBLOCK, 0o600, dir_fd=directory)
            stack.callback(os.close, lock)
            info = os.fstat(lock)
            if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                    or info.st_nlink != 1 or info.st_mode & 0o077):
                return {**result, "reason": "receipt_lock_invalid"}
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return {**result, "delivery_state": "in_progress", "reason": "receipt_busy"}

            lock_identity = (*directory_identity(info), info.st_nlink)

            def verify_namespace():
                for parent, part, opened, expected in directories:
                    if (directory_identity(os.fstat(opened)) != expected
                            or directory_identity(os.stat(part, dir_fd=parent,
                                                          follow_symlinks=False)) != expected):
                        raise ValueError("receipt_namespace_changed")
                fd_info = os.fstat(lock)
                path_info = os.stat(receipt.name + ".lock", dir_fd=directory,
                                    follow_symlinks=False)
                if ((*directory_identity(fd_info), fd_info.st_nlink) != lock_identity
                        or (*directory_identity(path_info), path_info.st_nlink) != lock_identity):
                    raise ValueError("receipt_lock_changed")

            verify_namespace()

            def pairs(rows):
                value = {}
                for key, item in rows:
                    if key in value:
                        raise ValueError("duplicate_receipt_key")
                    value[key] = item
                return value

            try:
                fd = os.open(receipt.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                             | os.O_CLOEXEC, dir_fd=directory)
            except FileNotFoundError:
                state = {**identity, "attempts": 0, "delivery_state": "prepared",
                         "reason": "new", "acknowledged_turn_id": None}
            else:
                try:
                    before = os.fstat(fd)
                    if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                            or before.st_nlink != 1 or before.st_mode & 0o077
                            or not 0 < before.st_size <= 4096):
                        raise ValueError("receipt_invalid")
                    raw = os.read(fd, 4097)
                    after = os.fstat(fd)
                    fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
                              "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns")
                    path_info = os.stat(receipt.name, dir_fd=directory, follow_symlinks=False)
                    if (len(raw) != before.st_size
                            or any(getattr(before, k) != getattr(after, k)
                                   or getattr(before, k) != getattr(path_info, k) for k in fields)):
                        raise ValueError("receipt_changed")
                    state = json.loads(raw, object_pairs_hook=pairs)
                finally:
                    os.close(fd)
                if (type(state) is not dict
                        or any(type(state.get(k)) is not type(v) or state.get(k) != v
                               for k, v in identity.items())
                        or set(state) != set(identity) | {"attempts", "delivery_state", "reason",
                                                         "acknowledged_turn_id"}
                        or type(state["attempts"]) is not int
                        or not 0 <= state["attempts"] <= max_attempts
                        or state["delivery_state"] not in {"prepared", "checking", "not_sent",
                                                          "delivery_unknown", "accepted", "refused"}
                        or (state["delivery_state"] == "accepted"
                            and state["acknowledged_turn_id"] != expected_turn_id)):
                    return {**result, "reason": "receipt_identity_mismatch"}

            def outcome():
                status = state["delivery_state"]
                return {**result, "delivery_state": status, "reason": state["reason"],
                        "attempts": state["attempts"],
                        "message_sha256": identity["message_sha256"],
                        "enqueue_acknowledged": status == "accepted",
                        "retryable": status == "not_sent" and state["attempts"] < max_attempts}

            def persist(status, reason, acknowledged=None):
                verify_namespace()
                updated = {**state, "delivery_state": status, "reason": reason,
                           "acknowledged_turn_id": acknowledged}
                data = (json.dumps(updated, sort_keys=True, separators=(",", ":")) + "\n").encode()
                if len(data) > 4096:
                    raise ValueError("receipt_size_bound")
                name = "." + receipt.name + "." + uuid.uuid4().hex
                try:
                    fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW
                                 | os.O_CLOEXEC, 0o600, dir_fd=directory)
                    try:
                        view = memoryview(data)
                        while view:
                            written = os.write(fd, view)
                            if written <= 0:
                                raise OSError("short_receipt_write")
                            view = view[written:]
                        os.fsync(fd)
                    finally:
                        os.close(fd)
                    os.replace(name, receipt.name, src_dir_fd=directory, dst_dir_fd=directory)
                    os.fsync(directory)
                    state.update(updated)
                finally:
                    with contextlib.suppress(FileNotFoundError):
                        os.unlink(name, dir_fd=directory)

            if state["delivery_state"] in {"accepted", "delivery_unknown", "refused"}:
                return outcome()
            if prior_delivery_state == "unknown":
                persist("delivery_unknown", "prior_channel_ambiguous")
                return outcome()
            if state["attempts"] >= max_attempts:
                persist("refused", "attempt_budget_exhausted")
                return outcome()
            state["attempts"] += 1
            persist("checking", "attempt_reserved")
            dispatching = False
            try:
                with _open_codex_app_connection(
                    str(endpoint), timeout_seconds=remaining(), max_response_bytes=4194304
                ) as connection:
                    response_bytes, notifications, request_number = 0, 0, 0

                    def send(value):
                        # The underlying Unix socket write must share the total deadline.
                        connection.socket.settimeout(remaining())
                        connection.send(json.dumps(value))

                    def rpc(method, params):
                        nonlocal response_bytes, notifications, request_number
                        request_number += 1
                        number = request_number
                        if method == "turn/steer":
                            verify_namespace()
                        send({"jsonrpc": "2.0", "id": number, "method": method, "params": params})
                        while True:
                            raw = connection.recv(timeout=remaining())
                            if type(raw) not in (str, bytes):
                                raise ValueError("rpc_frame_invalid")
                            response_bytes += len(raw.encode("utf-8") if type(raw) is str else raw)
                            if response_bytes > 4194304:
                                raise ValueError("rpc_response_bound")
                            reply = json.loads(raw, object_pairs_hook=pairs)
                            if type(reply) is not dict:
                                raise ValueError("rpc_reply_invalid")
                            if reply.get("id") == number:
                                if "error" in reply or "result" not in reply:
                                    raise ValueError("rpc_refused")
                                return reply["result"]
                            notifications += 1
                            if notifications > 32:
                                raise ValueError("rpc_notification_bound")

                    rpc("initialize", {"clientInfo": {"name": "ipfs-accelerate-handoff", "version": "1"}})
                    send({"jsonrpc": "2.0", "method": "initialized"})
                    thread = rpc("thread/read", {"threadId": thread_id, "includeTurns": False})
                    target = thread.get("thread") if type(thread) is dict else None
                    if (type(target) is not dict or target.get("id") != thread_id
                            or target.get("canAcceptDirectInput") is not True):
                        persist("refused", "target_not_accepting_input")
                        return outcome()
                    listing = rpc("thread/turns/list", {"threadId": thread_id, "limit": 1,
                                                        "sortDirection": "desc"})
                    rows = listing.get("data") if type(listing) is dict else None
                    row = rows[0] if type(rows) is list and len(rows) == 1 else None
                    turn = row.get("turn", row) if type(row) is dict else None
                    if (type(turn) is not dict or turn.get("id") != expected_turn_id
                            or turn.get("status") != "inProgress"):
                        persist("refused", "target_turn_changed")
                        return outcome()
                    # A crash or lost reply after this durable point must never replay.
                    persist("delivery_unknown", "steer_reserved")
                    dispatching = True
                    ack = rpc("turn/steer", {"threadId": thread_id,
                              "expectedTurnId": expected_turn_id,
                              "input": [{"type": "text", "text": message, "text_elements": []}]})
                    if type(ack) is not dict or ack.get("turnId") != expected_turn_id:
                        raise ValueError("steer_ack_mismatch")
                    persist("accepted", "enqueue_acknowledged", expected_turn_id)
                    return outcome()
            except Exception:
                if state["delivery_state"] in {"accepted", "refused"}:
                    return outcome()
                status = "delivery_unknown" if dispatching else "not_sent"
                reason = "steer_unacknowledged" if dispatching else "preflight_unavailable"
                try:
                    persist(status, reason)
                except (OSError, ValueError):
                    # The durable pre-steer reservation already forbids replay.
                    state.update(delivery_state=status, reason="receipt_update_failed")
                return outcome()
    except (OSError, ValueError, TypeError, UnicodeError):
        return {**result, "reason": "receipt_unavailable"}


class OpenAICodexCLIIntegration(BaseCLIWrapper):
    """
    OpenAI Codex CLI integration with common cache infrastructure.

    Command contract: ``codex exec`` (stdin prompt, ``--output-last-message``).
    """

    def __init__(
        self,
        codex_path: str = "codex",
        enable_cache: bool = True,
        cache: Optional[LLMAPICache] = None,
        *,
        command_override: Optional[Union[str, Sequence[str]]] = None,
        verify_on_init: bool = False,
        **kwargs: Any,
    ):
        """
        Initialize OpenAI Codex CLI integration.

        Args:
            codex_path: Path to ``codex`` executable (not the obsolete ``openai`` CLI)
            enable_cache: Whether to enable caching for non-side-effecting reads
            cache: Custom cache instance (uses LLM cache if None)
            command_override: Optional argv prefix (env override wins if unset)
            verify_on_init: When True, probe ``--version`` (default False)
            **kwargs: Additional arguments for BaseCLIWrapper
        """
        if cache is None:
            cache = get_llm_cache("openai_codex")

        override = parse_argv_override(command_override)
        if override is None:
            override = resolve_command_override_from_env(*_CODEX_CMD_ENV)

        # codex exec can write files / run tools depending on sandbox — treat
        # generation as side-effecting so generic cache/retry stay off.
        super().__init__(
            cli_path=codex_path if codex_path else _default_codex_path(),
            cache=cache,
            enable_cache=enable_cache,
            verify_on_init=verify_on_init,
            command_override=override,
            side_effecting_default=True,
            **kwargs,
        )

    def get_tool_name(self) -> str:
        return "OpenAI Codex CLI"

    def send_message_to_thread(self, **kwargs: Any) -> Dict[str, Any]:
        """Use the existing daemon handoff, without generation cache or retry."""
        return send_codex_thread_message(**kwargs)

    def generate_code(
        self,
        prompt: str,
        model: str = "gpt-6.1-sol",
        temperature: float = 0.0,
        *,
        sandbox: Optional[str] = None,
        skip_git_repo_check: bool = True,
        json_mode: bool = False,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Generate code from prompt via ``codex exec``.

        Args:
            prompt: Code generation prompt (passed on stdin, not shell-interpolated)
            model: Model to use (``-m``)
            temperature: Accepted for API compatibility; Codex CLI ignores it
            sandbox: Optional sandbox mode (omit / ``auto`` skips the flag)
            skip_git_repo_check: Pass ``--skip-git-repo-check`` (default True)
            json_mode: Pass ``--json`` when tracing is needed
            timeout: Command timeout override
            **kwargs: Additional arguments (ignored for command construction)

        Returns:
            Command result dict with generated code in ``stdout`` / ``text``
        """
        _ = temperature
        _ = kwargs
        env_sandbox = sandbox
        if env_sandbox is None:
            env_sandbox = os.getenv("ipfs_accelerate_py_CODEX_SANDBOX", "auto")

        with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as handle:
            last_msg_path = handle.name

        try:
            full_argv = build_codex_exec_argv(
                base_argv=self._base_argv(),
                model=model,
                prompt=prompt,
                last_message_path=last_msg_path,
                sandbox=env_sandbox,
                skip_git_repo_check=skip_git_repo_check,
                json_mode=json_mode,
            )
            # Side-effecting: no cache, no retry.
            response = self._run_command_with_retry(
                args=[],
                operation="codex_exec",
                stdin=str(prompt),
                timeout=timeout,
                side_effecting=True,
                full_argv=full_argv,
                prompt=prompt,
                model=model,
            )

            text_out = ""
            try:
                with open(last_msg_path, "r", encoding="utf-8", errors="replace") as fh:
                    text_out = fh.read().strip()
            except OSError:
                text_out = ""

            if text_out:
                response["stdout"] = text_out
                response["text"] = text_out
            else:
                response["text"] = (response.get("stdout") or "").strip()

            response["command_contract"] = "codex exec"
            response["model"] = model
            return response
        finally:
            try:
                os.unlink(last_msg_path)
            except OSError:
                pass


# Global instance
_global_openai_codex_cli: Optional[OpenAICodexCLIIntegration] = None


def get_openai_codex_cli_integration() -> OpenAICodexCLIIntegration:
    """Get or create the global OpenAI Codex CLI integration instance."""
    global _global_openai_codex_cli

    if _global_openai_codex_cli is None:
        _global_openai_codex_cli = OpenAICodexCLIIntegration()

    return _global_openai_codex_cli


def reset_openai_codex_cli_integration() -> None:
    """Test helper: drop the module-level singleton."""
    global _global_openai_codex_cli
    _global_openai_codex_cli = None


def read_codex_thread_state(
    *, socket_path, thread_id, timeout_seconds=15.0, max_response_bytes=262144,
) -> Dict[str, Any]:
    """Read bounded metadata through the existing Unix transport, without MCP.

    This is one connection to a caller-supplied existing owned socket. It never
    starts/resumes/steers a thread, discovers a daemon, or reads/writes a delivery
    receipt. The two metadata reads are not atomic. Limited or missing history
    cannot establish non-delivery, process cleanup, task ownership or authority.
    A valid thread observation survives failure of the subsequent latest-turn
    read as partial metadata. Connection close adds at most the client's bounded
    close timeout to the shared request deadline.
    """
    result = {
        "schema": "codex-existing-thread-state/v1",
        "observation_state": "refused", "reason": "invalid_request",
        "thread_id": None, "thread_snapshot": None, "latest_turn": None,
        "read_only": True, "snapshot_atomic": False, "history_complete": False,
        "delivery_assessed": False, "processing_confirmed": False,
        "completion_eligible": False, "execution_granted": False,
        "claim_granted": False,
    }
    try:
        if (type(thread_id) is not str
                or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", thread_id) is None
                or type(timeout_seconds) not in (int, float)
                or not math.isfinite(timeout_seconds) or not 0 < timeout_seconds <= 60
                or type(max_response_bytes) is not int
                or not 4096 <= max_response_bytes <= 4194304):
            return result
        endpoint = Path(socket_path)
        if (not endpoint.is_absolute() or ".." in endpoint.parts
                or "\0" in str(endpoint) or len(os.fsencode(endpoint)) > 107):
            return result
    except (TypeError, ValueError, UnicodeError):
        return result
    result["thread_id"] = thread_id
    deadline = time.monotonic() + timeout_seconds

    def remaining():
        left = deadline - time.monotonic()
        if left <= 0:
            raise TimeoutError("read_deadline")
        return left

    def pairs(rows):
        value = {}
        for key, item in rows:
            if key in value:
                raise ValueError("duplicate_rpc_key")
            value[key] = item
        return value

    def reject_constant(value):
        raise ValueError("nonfinite_rpc_number")

    try:
        with _open_codex_app_connection(
            str(endpoint), timeout_seconds=remaining(), max_response_bytes=max_response_bytes,
        ) as connection:
            response_bytes, notifications, request_number = 0, 0, 0

            def send(value):
                connection.socket.settimeout(remaining())
                connection.send(json.dumps(value, allow_nan=False))

            def rpc(method, params):
                nonlocal response_bytes, notifications, request_number
                request_number += 1
                number = request_number
                send({"jsonrpc": "2.0", "id": number, "method": method, "params": params})
                while True:
                    raw = connection.recv(timeout=remaining())
                    if type(raw) not in (str, bytes):
                        raise ValueError("rpc_frame_invalid")
                    response_bytes += len(raw.encode("utf-8") if type(raw) is str else raw)
                    if response_bytes > max_response_bytes:
                        raise ValueError("rpc_response_bound")
                    reply = json.loads(raw, object_pairs_hook=pairs, parse_constant=reject_constant)
                    if type(reply) is not dict:
                        raise ValueError("rpc_reply_invalid")
                    if "id" in reply:
                        if type(reply["id"]) is not int or reply["id"] != number:
                            raise ValueError("rpc_response_id_mismatch")
                        if "error" in reply or "result" not in reply:
                            raise ValueError("rpc_refused")
                        return reply["result"]
                    notifications += 1
                    if notifications > 32:
                        raise ValueError("rpc_notification_bound")

            rpc("initialize", {"clientInfo": {"name": "ipfs-accelerate-state-reader", "version": "1"}})
            send({"jsonrpc": "2.0", "method": "initialized"})
            thread_reply = rpc("thread/read", {"threadId": thread_id, "includeTurns": False})
            target = thread_reply.get("thread") if type(thread_reply) is dict else None
            if type(target) is not dict or target.get("id") != thread_id:
                return {**result, "reason": "thread_identity_mismatch"}
            accepts = target.get("canAcceptDirectInput")
            if accepts is not None and type(accepts) is not bool:
                return {**result, "reason": "invalid_thread_metadata"}
            status = target.get("status")
            if type(status) is dict:
                status = status.get("type")
            if status is not None and (type(status) is not str
                    or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", status) is None):
                return {**result, "reason": "invalid_thread_metadata"}
            result.update(
                observation_state="partial", reason="latest_turn_unavailable",
                thread_snapshot={"id": thread_id, "can_accept_direct_input": accepts, "status": status},
            )
            listing = rpc("thread/turns/list", {
                "threadId": thread_id, "limit": 1, "sortDirection": "desc",
            })
            rows = listing.get("data") if type(listing) is dict else None
            if type(rows) is not list or len(rows) > 1:
                return result
            latest = None
            if rows:
                row = rows[0]
                turn = row.get("turn", row) if type(row) is dict else None
                if (type(turn) is not dict or type(turn.get("id")) is not str
                        or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}", turn["id"]) is None
                        or type(turn.get("status")) is not str
                        or re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", turn["status"]) is None):
                    return result
                latest = {"id": turn["id"], "status": turn["status"]}
            result.update(observation_state="observed", reason="metadata_observed", latest_turn=latest)
            return result
    except Exception:
        if result["thread_snapshot"] is not None:
            return {**result, "observation_state": "partial", "reason": "latest_turn_unavailable"}
        return {**result, "observation_state": "unavailable", "reason": "read_transport_unavailable"}


def codex_thread_state_main(argv: Optional[Sequence[str]] = None) -> int:
    """Explicit metadata query; no delivery retry, model, daemon or receipt use."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Read existing Codex thread metadata through its owned Unix socket.")
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-response-bytes", type=int, default=262144)
    options = parser.parse_args(argv)
    fallback = {
        "schema": "codex-existing-thread-state/v1", "observation_state": "unavailable",
        "reason": "cli_read_unavailable", "thread_id": None, "thread_snapshot": None,
        "latest_turn": None, "read_only": True, "snapshot_atomic": False,
        "history_complete": False, "delivery_assessed": False, "processing_confirmed": False,
        "completion_eligible": False, "execution_granted": False, "claim_granted": False,
    }
    try:
        outcome = read_codex_thread_state(
            socket_path=options.socket_path, thread_id=options.thread_id,
            timeout_seconds=options.timeout_seconds, max_response_bytes=options.max_response_bytes,
        )
        if (type(outcome) is not dict or set(outcome) != set(fallback)
                or outcome.get("schema") != fallback["schema"]
                or outcome.get("observation_state") not in {"observed", "partial", "unavailable", "refused"}
                or outcome.get("read_only") is not True
                or any(outcome.get(key) is not False for key in (
                    "snapshot_atomic", "history_complete", "delivery_assessed", "processing_confirmed",
                    "completion_eligible", "execution_granted", "claim_granted",
                ))):
            outcome = fallback
        wire = json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n"
    except Exception:
        outcome = fallback
        wire = json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n"
    try:
        written = sys.stdout.write(wire)
        if type(written) is not int or written != len(wire):
            return 1
        sys.stdout.flush()
    except Exception:
        return 1
    return {"observed": 0, "partial": 1, "unavailable": 1, "refused": 2}[outcome["observation_state"]]


def read_codex_handoff_artifact(
    *, private_root, artifact_path, expected_bytes, expected_sha256, max_bytes=131072,
) -> Dict[str, Any]:
    """Verify one explicit private handoff, independent of messaging and RAM.

    No discovery, queue, receipt mutation, retry or interpretation is performed.
    The caller selects the artifact and its read budget. A matching digest proves
    these bytes only; it confers no claim, admission, execution, legal meaning or
    completion authority. The root must remain owned0700 for its lifetime. All
    path components are opened without following symlinks; namespace checks are
    cooperative and do not defend against an owner replacing it after return.
    """
    result = {
        "schema": "codex-private-handoff-artifact/v1", "artifact_state": "refused",
        "reason": "invalid_request", "bytes": None, "sha256": None,
        "full9": None, "payload_base64": None, "read_only": True,
        "delivery_assessed": False, "processing_confirmed": False,
        "completion_eligible": False, "execution_granted": False, "claim_granted": False,
    }
    try:
        if (type(expected_bytes) is not int or type(max_bytes) is not int
                or not 1 <= expected_bytes <= max_bytes <= 1048576
                or type(expected_sha256) is not str
                or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None):
            return result
        root, path = Path(private_root), Path(artifact_path)
        if (not root.is_absolute() or not path.is_absolute()
                or ".." in root.parts or ".." in path.parts
                or "\0" in str(root) or "\0" in str(path)
                or path.parent != root or len(root.parts) < 2):
            return result
    except (TypeError, ValueError, UnicodeError):
        return result
    keys = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
            "st_size", "st_mtime_ns", "st_ctime_ns")
    def full9(value):
        return {key: str(getattr(value, key)) for key in keys}
    def directory_identity(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)
    try:
        with contextlib.ExitStack() as stack:
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            stack.callback(os.close, directory)
            directories = []
            for part in root.parts[1:]:
                parent = directory
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                                    | os.O_CLOEXEC, dir_fd=parent)
                stack.callback(os.close, directory)
                directories.append((parent, part, directory, directory_identity(os.fstat(directory))))
            root_info = os.fstat(directory)
            if root_info.st_uid != os.geteuid() or stat.S_IMODE(root_info.st_mode) != 0o700:
                return {**result, "reason": "artifact_root_not_private"}
            def verify_namespace():
                for parent, part, opened, identity in directories:
                    if (directory_identity(os.fstat(opened)) != identity
                            or directory_identity(os.stat(part, dir_fd=parent, follow_symlinks=False)) != identity):
                        raise ValueError("artifact_namespace_changed")
            verify_namespace()
            fd = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
                         | os.O_CLOEXEC, dir_fd=directory)
            stack.callback(os.close, fd)
            before = os.fstat(fd)
            if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                    or before.st_nlink != 1 or before.st_size != expected_bytes):
                return {**result, "reason": "artifact_identity_mismatch"}
            chunks, count = [], 0
            while count <= expected_bytes:
                block = os.read(fd, min(65536, expected_bytes + 1 - count))
                if not block:
                    break
                chunks.append(block)
                count += len(block)
            body = b"".join(chunks)
            after = os.fstat(fd)
            current = os.stat(path.name, dir_fd=directory, follow_symlinks=False)
            verify_namespace()
            if count != expected_bytes or full9(before) != full9(after) or full9(before) != full9(current):
                return {**result, "reason": "artifact_changed"}
            digest = hashlib.sha256(body).hexdigest()
            if digest != expected_sha256:
                return {**result, "reason": "artifact_digest_mismatch"}
        import base64
        return {
            **result, "artifact_state": "verified", "reason": "exact_artifact_verified",
            "bytes": count, "sha256": digest, "full9": full9(before),
            "payload_base64": base64.b64encode(body).decode("ascii"),
        }
    except (OSError, ValueError, TypeError, UnicodeError):
        return {**result, "artifact_state": "unavailable", "reason": "artifact_unavailable"}


def codex_handoff_artifact_main(argv: Optional[Sequence[str]] = None) -> int:
    """Read one selected immutable private artifact; never enqueue work."""
    import argparse
    import sys
    parser = argparse.ArgumentParser(description="Verify a selected private handoff by exact bytes and SHA256.")
    parser.add_argument("--private-root", required=True)
    parser.add_argument("--artifact-path", required=True)
    parser.add_argument("--expected-bytes", type=int, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--max-bytes", type=int, default=131072)
    args = parser.parse_args(argv)
    outcome = read_codex_handoff_artifact(
        private_root=args.private_root, artifact_path=args.artifact_path,
        expected_bytes=args.expected_bytes, expected_sha256=args.expected_sha256,
        max_bytes=args.max_bytes,
    )
    wire = json.dumps(outcome, sort_keys=True, allow_nan=False) + "\n"
    try:
        written = sys.stdout.write(wire)
        if type(written) is not int or written != len(wire):
            return 1
        sys.stdout.flush()
    except Exception:
        return 1
    return {"verified": 0, "unavailable": 1, "refused": 2}[outcome["artifact_state"]]


def write_codex_handoff_artifact(
    *, private_root, artifact_path, payload, max_bytes=131072,
) -> Dict[str, Any]:
    """Retain exact caller bytes before delivery; never enqueue or grant work.

    The caller must retain its owned0700 namespace. An existing path is never
    overwritten or silently reused. Partial files survive failures for diagnosis.
    Namespace checks are cooperative, as for the existing artifact reader.
    """
    result = {
        "schema": "codex-private-handoff-artifact-write/v1",
        "artifact_state": "refused", "reason": "invalid_request",
        "path": None, "bytes": None, "sha256": None, "full9": None,
        "created": False, "readback_verified": False,
        "delivery_assessed": False, "processing_confirmed": False,
        "completion_eligible": False, "execution_granted": False,
        "claim_granted": False, "retryable": False,
    }
    try:
        if (type(payload) is not bytes or type(max_bytes) is not int
                or not 1 <= len(payload) <= max_bytes <= 1048576):
            return result
        root, path = Path(private_root), Path(artifact_path)
        if (not root.is_absolute() or not path.is_absolute()
                or ".." in root.parts or ".." in path.parts
                or "\0" in str(root) or "\0" in str(path)
                or path.parent != root or len(root.parts) < 2):
            return result
    except (TypeError, ValueError, UnicodeError):
        return result
    result.update(path=str(path), bytes=len(payload),
                  sha256=hashlib.sha256(payload).hexdigest())
    def directory_identity(value):
        return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)
    keys = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
            "st_size", "st_mtime_ns", "st_ctime_ns")
    def full9(value):
        return {key: str(getattr(value, key)) for key in keys}
    try:
        with contextlib.ExitStack() as stack:
            directory = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
            stack.callback(os.close, directory)
            directories = []
            for part in root.parts[1:]:
                parent = directory
                directory = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                                    | os.O_CLOEXEC, dir_fd=parent)
                stack.callback(os.close, directory)
                directories.append((parent, part, directory, directory_identity(os.fstat(directory))))
            info = os.fstat(directory)
            if info.st_uid != os.geteuid() or stat.S_IMODE(info.st_mode) != 0o700:
                return {**result, "reason": "artifact_root_not_private"}
            def verify_namespace():
                for parent, part, opened, identity in directories:
                    if (directory_identity(os.fstat(opened)) != identity
                            or directory_identity(os.stat(part, dir_fd=parent, follow_symlinks=False)) != identity):
                        raise ValueError("artifact_namespace_changed")
            verify_namespace()
            fd = os.open(path.name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600, dir_fd=directory)
            stack.callback(os.close, fd)
            result["created"] = True
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 1:
                raise ValueError("artifact_identity_mismatch")
            pending = memoryview(payload)
            while pending:
                count = os.write(fd, pending)
                if type(count) is not int or count <= 0 or count > len(pending):
                    raise OSError("artifact_write_incomplete")
                pending = pending[count:]
            os.fchmod(fd, 0o400)
            os.fsync(fd)
            os.fsync(directory)
            before = full9(os.fstat(fd))
            verify_namespace()
            verified = read_codex_handoff_artifact(
                private_root=str(root), artifact_path=str(path),
                expected_bytes=len(payload), expected_sha256=result["sha256"], max_bytes=max_bytes,
            )
            verify_namespace()
            if (verified.get("artifact_state") != "verified"
                    or verified.get("full9") != before
                    or full9(os.fstat(fd)) != before
                    or full9(os.stat(path.name, dir_fd=directory, follow_symlinks=False)) != before):
                raise ValueError("artifact_readback_or_identity_mismatch")
        return {**result, "artifact_state": "retained", "reason": "exact_artifact_retained",
                "full9": before, "readback_verified": True}
    except FileExistsError:
        return {**result, "reason": "artifact_already_exists"}
    except Exception as error:
        # No cleanup/retry: a created file may contain some or all caller bytes.
        return {**result, "artifact_state": "incomplete" if result["created"] else "refused",
                "reason": "artifact_not_retained", "error_type": type(error).__name__}


def codex_thread_handoff_main(argv: Optional[Sequence[str]] = None) -> int:
    """Explicit existing-thread handoff command; acknowledgement is enqueue only.

    This entry point delegates once to the existing durable transport. It does
    not initialize a generation wrapper, discover/start a daemon, create a
    receipt namespace, or retry. Callers must retain the supplied private
    receipt namespace and classify any earlier possibly-sent channel as unknown.
    """
    import argparse
    import sys

    read_arguments = list(sys.argv[1:] if argv is None else argv)
    if read_arguments and read_arguments[0] == "read-thread":
        return codex_thread_state_main(read_arguments[1:])
    if read_arguments and read_arguments[0] == "read-artifact":
        return codex_handoff_artifact_main(read_arguments[1:])

    parser = argparse.ArgumentParser(
        description="Bounded handoff to an existing Codex thread; no model or daemon starts.",
        epilog=(
            "Keep the owned private receipt directory and lock for this operation's lifetime. "
            "If an earlier channel may have sent, select prior delivery state 'unknown'. "
            "An enqueue acknowledgement grants no processing, claim, or execution authority."
        ),
    )
    parser.add_argument("command", choices=("handoff",))
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--thread-id", required=True)
    parser.add_argument("--expected-turn-id", required=True)
    parser.add_argument("--message", required=True)
    parser.add_argument("--receipt-path", required=True)
    parser.add_argument(
        "--prior-delivery-state", required=True,
        choices=("not_attempted", "not_sent", "unknown"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--offline-private-root")
    parser.add_argument("--offline-artifact-path")
    options = parser.parse_args(argv)
    if (options.offline_private_root is None) != (options.offline_artifact_path is None):
        parser.error("offline private root and artifact path must be supplied together")

    def unknown(reason):
        return {
            "delivery_state": "delivery_unknown", "reason": reason,
            "enqueue_acknowledged": False, "processing_confirmed": False,
            "completion_eligible": False, "execution_granted": False,
            "claim_granted": False, "retryable": False,
        }

    offline_artifact = None
    if options.offline_private_root is not None:
        try:
            offline_artifact = write_codex_handoff_artifact(
                private_root=options.offline_private_root,
                artifact_path=options.offline_artifact_path,
                payload=options.message.encode("utf-8"),
            )
        except Exception:
            offline_artifact = {"artifact_state": "incomplete", "reason": "cli_artifact_exception"}
    if offline_artifact is not None and (
            type(offline_artifact) is not dict or offline_artifact.get("artifact_state") != "retained"):
        # No send by this call does not prove that an earlier attempt did not
        # deliver. The existing receipt is deliberately not inspected here.
        outcome = unknown("offline_artifact_not_retained")
        outcome["current_call_transport_attempted"] = False
    else:
        try:
            outcome = send_codex_thread_message(
                socket_path=options.socket_path,
                thread_id=options.thread_id,
                expected_turn_id=options.expected_turn_id,
                message=options.message,
                receipt_path=options.receipt_path,
                timeout_seconds=options.timeout_seconds,
                max_attempts=options.max_attempts,
                prior_delivery_state=options.prior_delivery_state,
            )
        except Exception:
            # The transport may have sent before an unexpected failure. Never
            # convert that uncertainty into permission for another delivery.
            outcome = unknown("cli_transport_exception")

    exit_codes = {
        "accepted": 0, "not_sent": 1, "refused": 2,
        "delivery_unknown": 3, "in_progress": 4,
    }
    if (
        type(outcome) is not dict
        or type(outcome.get("delivery_state")) is not str
        or outcome["delivery_state"] not in exit_codes
        or any(outcome.get(key) is not False for key in (
            "processing_confirmed", "completion_eligible", "execution_granted", "claim_granted",
        ))
        or outcome.get("enqueue_acknowledged") is not (outcome["delivery_state"] == "accepted")
        or type(outcome.get("retryable")) is not bool
        or (outcome["delivery_state"] != "not_sent" and outcome["retryable"] is not False)
    ):
        outcome = unknown("cli_invalid_result")
    if offline_artifact is not None:
        outcome = {**outcome, "offline_artifact": offline_artifact}
    try:
        wire = json.dumps(outcome, sort_keys=True, allow_nan=False)
    except Exception:
        outcome = unknown("cli_invalid_result")
        if offline_artifact is not None:
            outcome["offline_artifact"] = offline_artifact if type(offline_artifact) is dict else {"artifact_state": "incomplete"}
        wire = json.dumps(outcome, sort_keys=True, allow_nan=False)
    # If output is lost, the caller must reuse the durable receipt rather than
    # mint another identity. Do not retry or clean the namespace here.
    try:
        line = wire + "\n"
        written = sys.stdout.write(line)
        if type(written) is not int or written != len(line):
            return 3
        sys.stdout.flush()
    except Exception:
        # The message may already be enqueued. Do not write again or replay;
        # the retained receipt remains the authority for delivery state.
        return 3
    return exit_codes[outcome["delivery_state"]]


if __name__ == "__main__":
    raise SystemExit(codex_thread_handoff_main())
