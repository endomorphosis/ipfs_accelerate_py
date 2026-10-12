"""Explicit matched raw/compact source views over the canonical worker checkout.

Guidance is optional. Hash bindings establish dispatch, never helper adoption,
behavioral equivalence, token savings, or task completion.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import re
import shlex
import sys
import uuid

SCHEMA = "supervisor-source-view-policy@1"
POLICIES = ("legacy", "source-view-raw@1", "source-view-compact@1")


def validate_policy(value: str, *, purpose: str, provider: str) -> str:
    if type(value) is not str or value not in POLICIES:
        raise ValueError("explicit supported source view policy required")
    if value != "legacy" and (purpose != "coding" or provider != "codex_cli"):
        raise ValueError("source view requires a Codex coding route")
    return value


def _path(value):
    if (type(value) is not str or not 1 <= len(value.encode()) <= 4096
            or any(ord(char) < 32 for char in value)):
        raise ValueError("bounded absolute source helper path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("canonical absolute source helper path required")
    return path


def _instruction(binding: dict) -> str:
    command = " ".join(shlex.quote(binding[key]) for key in ("python_executable", "helper_script"))
    command += " --root " + shlex.quote(binding["workspace"])
    command += " --audit-store " + shlex.quote(binding["audit_store"])
    representation = "raw" if binding["requested_policy"] == "source-view-raw@1" else "compact"
    return (
        "Optional source views and hash-checked edits:\n"
        "For explicit visible Python files, inspect symbols with the helper; execution uses the original checkout.\n"
        f"{command} index PATH...\n"
        f"{command} read PATH --symbol NAME --format {representation}\n"
        "Use symbol module for whole-file reads. Source views are bounded; retrieve necessary original detail "
        "with normal tools if unavailable. They do not prove behavior. To replace a supported symbol, "
        "write the complete replacement definition to a visible relative file, then use the source_sha256 from your read:\n"
        f"{command} replace PATH --symbol NAME --expected-sha256 HASH --replacement-file PATH --format {representation}\n"
        "Stale edits are rejected. Inspect the canonical diff, refresh changed views, and run the required checks. "
        "This does not expand allowed files or establish task completion. Helper use and token savings are measured separately.\n\n"
    )


def create_binding(*, workspace: Path, requested_policy: str) -> dict:
    validate_policy(requested_policy, purpose="coding", provider="codex_cli")
    if requested_policy == "legacy":
        raise ValueError("legacy policy has no source view binding")
    helper = Path(__file__).with_name("code_source_view.py").resolve(strict=True)
    if not helper.is_file() or helper.stat().st_size > 256_000:
        raise ValueError("bounded source helper source required")
    binding = {
        "schema": SCHEMA, "requested_policy": requested_policy,
        "python_executable": str(Path(sys.executable).absolute()),
        "helper_script": str(helper), "helper_sha256": hashlib.sha256(helper.read_bytes()).hexdigest(),
        "workspace": str(workspace),
        "audit_store": "/tmp/supervisor-source-view-" + uuid.uuid4().hex + "/operations.jsonl",
        "native_tool_interception": False, "helper_adoption_guaranteed": False,
        "token_savings_measured": False, "execution_authority": False, "completion_authority": False,
    }
    instruction = _instruction(binding)
    binding.update(instruction_sha256=hashlib.sha256(instruction.encode()).hexdigest(),
                   instruction_bytes=len(instruction.encode()))
    validate_binding(binding, workspace=workspace)
    return binding


def validate_binding(binding: dict, *, workspace: Path) -> str:
    fields = {"schema", "requested_policy", "python_executable", "helper_script", "helper_sha256",
              "workspace", "audit_store", "native_tool_interception", "helper_adoption_guaranteed",
              "token_savings_measured", "execution_authority", "completion_authority",
              "instruction_sha256", "instruction_bytes"}
    if (type(binding) is not dict or set(binding) not in (fields, fields | {"usage"})
            or binding.get("schema") != SCHEMA
            or type(binding.get("requested_policy")) is not str
            or binding["requested_policy"] not in POLICIES[1:]
            or any(binding.get(key) is not False for key in ("native_tool_interception", "helper_adoption_guaranteed",
                "token_savings_measured", "execution_authority", "completion_authority"))):
        raise ValueError("invalid source view policy binding")
    python, helper, recorded_workspace, audit = (_path(binding[key]) for key in
        ("python_executable", "helper_script", "workspace", "audit_store"))
    if (helper.name != "code_source_view.py" or recorded_workspace != workspace
            or helper.is_relative_to(workspace) or audit.is_relative_to(workspace)
            or re.fullmatch(r"/tmp/supervisor-source-view-[0-9a-f]{32}/operations.jsonl", str(audit)) is None
            or type(binding["helper_sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", binding["helper_sha256"]) is None):
        raise ValueError("invalid source view helper identity")
    instruction = _instruction(binding)
    if (type(binding["instruction_bytes"]) is not int
            or binding["instruction_bytes"] != len(instruction.encode())
            or binding["instruction_bytes"] > 8192
            or binding["instruction_sha256"] != hashlib.sha256(instruction.encode()).hexdigest()):
        raise ValueError("source view instruction binding differs")
    return instruction


def create_audit_store(binding: dict) -> tuple[int, int]:
    """Create a private invocation directory outside the task output tree."""
    import os
    directory = Path(binding["audit_store"]).parent
    directory.mkdir(mode=0o700)
    info = directory.lstat()
    return info.st_dev, info.st_ino


def finish_audit_store(binding: dict, *, workspace: Path, expected_identity) -> dict:
    """Export bounded content-free observations and remove only the known log."""
    import os
    import stat
    usage = {"status": "unavailable", "helper_operations_observed": None,
             "store_cleanup_complete": False, "artifacts_retained_after_invocation": None}
    if expected_identity is None:
        return usage
    audit = Path(binding["audit_store"])
    descriptor = None
    try:
        descriptor = os.open(audit.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(descriptor)
        if ((info.st_dev, info.st_ino) != expected_identity or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("source view audit directory identity changed")
        try:
            from .code_source_view import summarize_audit_store
            usage = summarize_audit_store(audit, expected_root=workspace)
            usage["helper_operations_observed"] = (bool(usage["operations"])
                if usage.get("status") in {"observed", "not_observed"}
                and type(usage.get("operations")) is int else None)
        except Exception:
            pass
        # Never recurse through worker-written files. unlink does not follow links.
        try:
            os.unlink(audit.name, dir_fd=descriptor)
        except FileNotFoundError:
            pass
        current = audit.parent.lstat()
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ValueError("source view audit directory identity changed")
        audit.parent.rmdir()
        usage.update(store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    except Exception:
        usage.update(store_cleanup_complete=False, artifacts_retained_after_invocation=None,
                     store_cleanup_error="cleanup_unavailable")
    finally:
        if descriptor is not None:
            os.close(descriptor)
    return usage
