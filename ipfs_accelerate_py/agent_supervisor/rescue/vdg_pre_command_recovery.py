"""Explicit, source-pinned VDG acquisition preflight; no completion authority.

Only the reviewed VDG-039 producer is eligible. Configuration lives in the
existing VDG lane manifest; ordinary lanes and every other command are inert.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

MANIFEST = Path("data/agent_supervisor/virtual_desktop_gui_mcp/watchdog/bundle_lanes.json")
TASK_ID = "VDG-039-R01"
SCRIPT = "swissknife/scripts/recover-vdg-capture-deadline.cjs"
PRODUCER = (
    "node", "swissknife/scripts/run-with-vdg-openrouter-catalog-input.mjs", "--",
    "node", "swissknife/scripts/run-vdg-all-families.mjs", "--timeout-seconds", "22200",
)
SCHEMA = "ipfs_accelerate_py.agent_supervisor.vdg_pre_command_recovery.v1"


class RecoveryPolicyError(ValueError):
    pass


def _sha(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


_LOADED_MODULE_SHA256 = _sha(Path(__file__))


def recovery_contract(command: Sequence[str], *, repo_root: Path) -> dict[str, Any] | None:
    """Stable binding including malformed policy and changed implementation.

    Policy errors must become normal producer failures, not unwind the lane
    while it is merely computing an attempt contract. Runtime identities and
    unrelated manifest timestamps never form part of this binding.
    """
    if tuple(command) != PRODUCER:
        return None
    root = repo_root.resolve()
    path = root / MANIFEST
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        return None
    except OSError as exc:
        return {"policy_error": type(exc).__name__, "module_sha256": _sha(Path(__file__))}
    try:
        manifest = json.loads(raw)
        lanes = manifest["lanes"]
        if not isinstance(lanes, list):
            raise ValueError("lanes must be a list")
        matches = [lane for lane in lanes if isinstance(lane, dict) and lane.get("bundle_key") == "vdg-planning"]
        if len(matches) != 1:
            raise ValueError("exactly one VDG lane required")
        policy = matches[0].get("pre_command_recovery")
        if policy is None:
            return None
    except (ValueError, TypeError, KeyError) as exc:
        return {"policy_error": type(exc).__name__, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
                "module_sha256": _sha(Path(__file__))}
    script = root / SCRIPT
    return {"policy": policy, "module_sha256": _sha(Path(__file__)),
            "loaded_module_sha256": _LOADED_MODULE_SHA256,
            "script_sha256": _sha(script), "script_is_symlink": script.is_symlink()}


def recovery_commands(task_id: str, command: Sequence[str], *, repo_root: Path,
                      apply: bool = True) -> list[tuple[list[str], float]]:
    """Validate everything before returning either transport's owned command."""
    if task_id != TASK_ID or tuple(command) != PRODUCER:
        return []
    binding = recovery_contract(command, repo_root=repo_root)
    if binding is None:
        return []
    if "policy_error" in binding:
        raise RecoveryPolicyError("VDG pre-command recovery manifest is invalid")
    policy = binding["policy"]
    keys = {"schema", "task_id", "producer_argv", "script", "script_sha256", "module_sha256", "transports", "timeout_seconds",
            "require_uninterrupted_after_producer"}
    if not isinstance(policy, dict) or set(policy) != keys:
        raise RecoveryPolicyError("VDG pre-command recovery policy fields differ")
    if (policy["schema"] != SCHEMA or policy["task_id"] != TASK_ID
            or policy["producer_argv"] != list(PRODUCER) or policy["script"] != SCRIPT
            or policy["transports"] != ["http", "libp2p"]
            or policy["require_uninterrupted_after_producer"] is not True):
        raise RecoveryPolicyError("VDG pre-command recovery scope differs")
    for name in ("script_sha256", "module_sha256"):
        if (not isinstance(policy[name], str) or not re.fullmatch(r"[a-f0-9]{64}", policy[name])
                or binding[name] != policy[name]):
            raise RecoveryPolicyError("VDG pre-command recovery source pin differs: " + name)
    if binding["loaded_module_sha256"] != binding["module_sha256"]:
        raise RecoveryPolicyError("VDG pre-command recovery loaded source differs; reload required")
    root = repo_root.resolve()
    script = root / SCRIPT
    if binding["script_is_symlink"] or not script.is_file() or not script.resolve().is_relative_to(root):
        raise RecoveryPolicyError("VDG pre-command recovery script is not an in-repository regular source")
    timeout = policy["timeout_seconds"]
    if type(timeout) is not int or not 1 <= timeout <= 300:
        raise RecoveryPolicyError("VDG pre-command recovery timeout must be 1..300 seconds")
    pid = os.getpid()
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        ticks = raw[raw.rfind(")") + 2:].split()[19]
    except (OSError, IndexError) as exc:
        raise RecoveryPolicyError("VDG pre-command recovery owner identity unavailable") from exc
    return [(["node", str(script), "--project-root", str(root / "swissknife"),
              "--family", "vdg-local", "--transport", transport, "--apply" if apply else "--require-clean",
              "--owner-pid", str(pid), "--owner-start-ticks", ticks, "--task-id", task_id], float(timeout))
            for transport in policy["transports"]]


def recovery_environment() -> dict[str, str]:
    """Match the reviewed producer wrapper before it has had a chance to run.

    An older watchdog may omit this newly declared flag. The wrapper already
    sets an absent value to 1 and refuses a conflicting explicit value; the
    preflight must see those same capture inputs without modifying its owner.
    """
    env = os.environ.copy()
    name = "VITE_VDG_OPENROUTER_CATALOG_SNAPSHOT"
    if env.get(name) not in (None, "1"):
        raise RecoveryPolicyError("VDG pre-command recovery conflicts with the declared snapshot mode")
    env[name] = "1"
    return env
