"""Campaign worker adapter for llm_router's public explicit Grok/Codex route.

Portal owns the task lease, workspace, deadline and validation. This module
only binds its trusted task metadata to the existing provider_fallback_runner.
It does not export or weaken the daemon's separately sealed route policy.
"""
from __future__ import annotations

import argparse
import codecs
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections.abc import Mapping
from typing import Any

from ...llm_router import build_codex_cli_command, find_grok_cli, probe_grok_codex_agent_route_readiness

MODEL = "grok-4.7"
POLICY = "grok_quota_auth_or_unavailable"
MODULE = "ipfs_accelerate_py.agent_supervisor.project_adapters.worldwide_legal_search_provider"
_TASK_RE = re.compile(r"WWL-(?:[0-9]{3}|SRC-[0-9a-f]{16}-(?:normalize|index|geo|publish|verify)-[0-9]{4}|REPAIR-(?:[0-9a-f]{16}|[0-9a-f]{64}))")


def _activity_binding(config: Mapping[str, Any], task_id: str, attempt: int) -> dict[str, Any] | None:
    """Read optional owner-scoped observation metadata, never launch authority."""
    try:
        root = Path(str(config["state_root"])).resolve()
        descriptor = os.open(root / "provider-activity-binding.json", os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            observed = os.fstat(source.fileno())
            if not stat.S_ISREG(observed.st_mode) or observed.st_uid != os.getuid() or observed.st_nlink != 1 or observed.st_mode & 0o077 or observed.st_size > 16384:
                return None
            binding = json.loads(source.read(16385))
        if not isinstance(binding, dict):
            return None
        body = {key: value for key, value in binding.items() if key != "binding_id"}
        identity = "sha256:" + hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        native_id = str(binding.get("native_attempt_id") or "")
        expected_path = root / "provider-activity" / (hashlib.sha256(native_id.encode()).hexdigest() + ".json")
        if (binding.get("schema") != "wwl.native-provider-activity-binding@1" or binding.get("binding_id") != identity
            or binding.get("run_id") != config.get("run_id", "worldwide-legal-search-v1") or binding.get("task_id") != task_id
            or binding.get("portal_attempt_number") != attempt or binding.get("portal_task_cid") != os.environ.get("IPFS_ACCELERATE_AGENT_TASK_CID")
            or not native_id or len(native_id) > 256 or not binding.get("task_cid")
            or re.fullmatch(r"[0-9a-f]{64}", str(binding.get("capability_nonce") or "")) is None
            or Path(str(binding.get("activity_path") or "")) != expected_path):
            return None
        issued = binding.get("issued_at", binding.get("started_at"))
        if type(issued) not in (int, float) or not math.isfinite(issued) or not -15 <= time.time() - issued <= int(config.get("task_timeout_seconds", 5400)) + 120:
            return None
        return binding
    except (OSError, ValueError, TypeError, KeyError):
        return None


class _SemanticActivity:
    """Count matched completed tools; prose, bytes and repeated frames do not count."""

    def __init__(self, binding: Mapping[str, Any]):
        self.binding = dict(binding)
        self.pending: dict[str, str] = {}
        self.completed: set[str] = set()
        self.sequence = 0
        self.buffer = bytearray()
        self.overlong = False

    def event(self, event: Any) -> None:
        if not isinstance(event, dict):
            return
        kind = event.get("type")
        tool_id = event.get("toolCallId")
        tool = event.get("toolName", event.get("title", "other"))
        started = kind == "tool_call" and event.get("status") in ("pending", "in_progress")
        finished = kind == "tool_call_update" and event.get("status") == "completed"
        if kind in ("item.started", "item.completed"):
            item = event.get("item", {})
            if not isinstance(item, dict) or item.get("type") not in ("command_execution", "file_change", "mcp_tool_call", "web_search"):
                return
            tool_id, tool = item.get("id"), item["type"]
            started, finished = kind == "item.started", kind == "item.completed"
        if not isinstance(tool_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,256}", tool_id):
            return
        if started and tool_id not in self.completed and len(self.pending) < 1024:
            self.pending[tool_id] = str(tool)[:100]
        if not finished or tool_id not in self.pending or tool_id in self.completed or len(self.completed) >= 100000:
            return
        self.pending.pop(tool_id)
        self.completed.add(tool_id)
        self.sequence += 1
        self.publish(tool_id)

    def publish(self, tool_id: str) -> None:
        binding = self.binding
        payload = {"schema": "wwl.native-provider-activity@1", **{key: binding[key] for key in
            ("run_id", "task_id", "task_cid", "portal_task_cid", "native_attempt_id", "portal_attempt_number", "binding_id", "capability_nonce")},
            "sequence": self.sequence, "event_kind": "tool_completed", "observed_at": time.time(),
            "tool_id_sha256": hashlib.sha256(tool_id.encode()).hexdigest(), "completion_authority": False}
        path = Path(binding["activity_path"])
        temporary = ""
        try:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".provider-activity-", dir=path.parent)
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                json.dump(payload, output, sort_keys=True)
                output.write("\n")
            os.replace(temporary, path)
        except OSError:
            pass  # Optional observation cannot alter routing or execution.
        finally:
            if temporary:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def feed(self, chunk: bytes, *, final: bool = False) -> None:
        for part in chunk.splitlines(keepends=True):
            ended = part.endswith(b"\n")
            if not self.overlong and len(self.buffer) + len(part) <= 1024 * 1024:
                self.buffer.extend(part)
            else:
                self.overlong = True
            if ended:
                if not self.overlong:
                    try:
                        self.event(json.loads(self.buffer))
                    except (ValueError, TypeError, RecursionError):
                        pass
                self.buffer.clear()
                self.overlong = False
        if final and self.buffer:
            self.feed(b"\n")


def _relay_with_activity(command: list[str], activity: _SemanticActivity) -> int:
    """Preserve prompt/custody and route output while observing bounded JSON."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE)
    assert process.stdout is not None
    binary_output = getattr(sys.stdout, "buffer", None)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    try:
        while chunk := process.stdout.read1(64 * 1024):
            if binary_output is not None:
                binary_output.write(chunk)
                binary_output.flush()
            else:
                sys.stdout.write(decoder.decode(chunk))
                sys.stdout.flush()
            activity.feed(chunk)
        activity.feed(b"", final=True)
        return int(process.wait())
    finally:
        process.stdout.close()


def implementation_command(python_executable: str, config_path: Path | str) -> str:
    """Return the native Portal command; prompt arrives on stdin in task cwd."""
    return shlex.join([str(python_executable), "-m", MODULE, "--config", str(Path(config_path).absolute())])


def _validate_config(config: Mapping[str, Any]) -> Mapping[str, Any] | None:
    provider = config.get("provider", {})
    if not isinstance(provider, Mapping):
        raise ValueError("campaign provider configuration must be an object")
    if provider.get("grok_model", MODEL) != MODEL or provider.get("fallback_policy", POLICY) != POLICY:
        raise ValueError("campaign requires exact grok-4.7 and canonical emergency policy")
    plan_path = config.get("plan_path")
    if plan_path:
        plan = json.loads(Path(str(plan_path)).read_text(encoding="utf-8"))
        route = plan.get("routing", {})
        if route.get("primary_provider") != "grok_cli" or route.get("primary_model") != MODEL or route.get("policy") != POLICY or route.get("fallback_provider") != "codex_cli":
            raise ValueError("campaign plan route differs from admitted Grok/Codex route")
        return plan
    return None


def _provider_settings(config: Mapping[str, Any]) -> dict[str, Any]:
    plan = _validate_config(config)
    provider = dict(config.get("provider", {}))
    return {
        # A missing primary binary is an eligible native pre-dispatch failure.
        # Preserve an executable name for the canonical readiness probe rather
        # than failing locally before it can select emergency Codex.
        "grok_bin": str(provider.get("grok_bin") or find_grok_cli() or "grok"),
        "codex_bin": str(provider.get("codex_bin") or shutil.which("codex") or ""),
        "codex_model": str(provider.get("codex_model") or os.environ.get("IPFS_ACCELERATE_AGENT_CODEX_MODEL") or "gpt-6.1-sol"),
        "codex_reasoning_effort": str(provider.get("codex_reasoning_effort") or os.environ.get("IPFS_ACCELERATE_AGENT_CODEX_REASONING_EFFORT") or "high"),
        "max_turns": int(provider.get("max_turns", 100000)),
        "codex_sandbox": str(provider.get("codex_sandbox", "workspace-write")),
        "plan": plan,
    }


def route_receipt_path(config: Mapping[str, Any], task_id: str, attempt: int) -> Path:
    """Use the exact native protected-provider receipt storage layout."""
    from ..validation.validation_runtime import PROOF_REUSE_STATE_ROOT_ENV, PROVIDER_PROTECTED_STATE_ROOT_ENV
    protected = os.environ.get(PROVIDER_PROTECTED_STATE_ROOT_ENV, "").strip()
    exposed = os.environ.get(PROOF_REUSE_STATE_ROOT_ENV, "").strip()
    if protected and exposed and Path(protected).resolve() != Path(exposed).resolve():
        raise ValueError("native protected provider state roots disagree")
    root = Path(protected or exposed or str(config["state_root"])).resolve()
    return root / "state" / "ptr_lane_worldwide_legal_search" / "provider_route_receipts" / task_id / f"provider-route-attempt-{attempt}.json"


def provider_readiness(config: Mapping[str, Any]) -> dict[str, Any]:
    """Probe model/auth availability without dispatching a model task."""
    settings = _provider_settings(config)
    result = probe_grok_codex_agent_route_readiness(
        grok_bin=settings["grok_bin"], codex_bin=settings["codex_bin"], grok_model=MODEL,
        codex_model=settings["codex_model"], codex_reasoning_effort=settings["codex_reasoning_effort"],
    )
    return {key: getattr(value, "value", value) for key, value in asdict(result).items()}


def _workspace(config: Mapping[str, Any], workspace: Path) -> Path:
    resolved = workspace.resolve(strict=True)
    roots = config.get("allowed_workspace_roots") or [str(Path(str(config["state_root"])) / "worktrees")]
    if not isinstance(roots, list) or not roots or not all(isinstance(value, str) and value for value in roots):
        raise ValueError("campaign needs explicit allowed task workspace roots")
    permitted = [Path(value).expanduser().resolve() for value in roots]
    if not any(resolved.is_relative_to(root) for root in permitted):
        raise ValueError("task workspace is outside the admitted campaign worktree roots")
    if not resolved.is_dir():
        raise ValueError("task workspace is not a directory")
    return resolved


def build_provider_argv(
    config: Mapping[str, Any], *, workspace: Path, task_id: str, attempt: int,
    python_executable: str | None = None,
) -> list[str]:
    """Build the canonical no-shell route command, without running providers."""
    if not _TASK_RE.fullmatch(task_id) or type(attempt) is not int or attempt < 1:
        raise ValueError("provider invocation needs a native task identity and positive attempt")
    settings = _provider_settings(config)
    if settings["plan"] is not None and task_id not in {str(task["id"]) for task in settings["plan"].get("tasks", [])}:
        raise ValueError("provider task identity is absent from the admitted campaign plan")
    if not settings["grok_bin"] or not settings["codex_bin"]:
        raise ValueError("campaign route requires available Grok and emergency Codex binaries")
    if settings["max_turns"] < 1:
        raise ValueError("Grok max-turns must be positive")
    workspace = _workspace(config, workspace)
    python = str(python_executable or sys.executable)
    supervisor_root = Path(__file__).resolve().parents[1]
    # This exact packaged adapter is recognized by provider_fallback_runner,
    # which gives it the private trusted failure-receipt descriptor. A raw
    # Grok process or arbitrary stderr cannot mint emergency fallback evidence.
    primary = [python, str(supervisor_root / "grok_cli_runner.py"),
               "--workspace", str(workspace), "--grok-bin", settings["grok_bin"],
               "--model", MODEL, "--mode", "agent", "--max-turns", str(settings["max_turns"])]
    fallback = build_codex_cli_command(
        codex_bin=settings["codex_bin"], workspace=workspace,
        model_name=settings["codex_model"], reasoning_effort=settings["codex_reasoning_effort"],
        sandbox=settings["codex_sandbox"], json_mode=True,
    )
    return [python, str(supervisor_root / "provider_fallback_runner.py"),
            "--workspace", str(workspace), "--primary-provider", "grok", "--fallback-provider", "codex",
            "--primary-command-json", json.dumps(primary, separators=(",", ":")),
            "--fallback-command-json", json.dumps(fallback, separators=(",", ":")),
            "--fallback-policy", POLICY, "--probe-route-readiness",
            "--probe-grok-bin", settings["grok_bin"], "--probe-codex-bin", settings["codex_bin"],
            "--probe-grok-model", MODEL, "--probe-codex-model", settings["codex_model"],
            "--probe-codex-reasoning-effort", settings["codex_reasoning_effort"],
            "--route-receipt-path", str(route_receipt_path(config, task_id, attempt)),
            "--route-task-id", task_id, "--route-attempt", str(attempt), "--route-stage", "implementation"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--readiness", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.readiness:
            print(json.dumps(provider_readiness(config), sort_keys=True))
            return 0
        task_id = os.environ.get("IPFS_ACCELERATE_AGENT_TASK_ID", "")
        attempt = int(os.environ.get("IPFS_ACCELERATE_AGENT_TASK_ATTEMPT", "0"))
        command = build_provider_argv(config, workspace=Path.cwd(), task_id=task_id, attempt=attempt)
        activity = _activity_binding(config, task_id, attempt)
        route_receipt_path(config, task_id, attempt).parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        print(json.dumps({"schema": "worldwide-legal-search.provider-invocation@1", "task_id": task_id, "attempt": attempt,
                          "primary_provider": "grok_cli", "primary_model": MODEL, "fallback_provider": "codex_cli",
                          "fallback_policy": POLICY, "completion_authority": False}), file=sys.stderr, flush=True)
        # Preserve the exact prompt stdin and native parent's process custody.
        # The canonical adapter replays sanitized provider streams and enforces
        # unchanged-workspace/known-pre-effect emergency eligibility.
        if activity is not None:
            return _relay_with_activity(command, _SemanticActivity(activity))
        os.execv(command[0], command)
    except (OSError, ValueError, KeyError) as exc:
        print(json.dumps({"schema": "worldwide-legal-search.provider-error@1", "error": str(exc), "completion_authority": False}), file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
