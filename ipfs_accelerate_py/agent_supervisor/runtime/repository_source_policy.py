"""Invocation-scoped native repository tools with separately observed adoption.

The binding configures a fixed stdio server. It does not intercept terminal
commands, guarantee tool use, or confer proof or completion authority.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import uuid

SCHEMA = "supervisor-repository-source-policy@1"
POLICIES = ("legacy", "repository-source-raw@1", "repository-source-compact@1")
TOOLS = ("repository_status", "repository_list", "repository_symbols", "repository_read",
         "repository_replace_symbol", "repository_refresh")
_FALSE_FIELDS = ("native_tool_interception", "helper_adoption_guaranteed", "token_savings_measured",
                 "proof_authority", "completion_authority")


def validate_policy(value, *, purpose, provider, source_view_policy="legacy", census_only=False):
    if type(value) is not str or value not in POLICIES:
        raise ValueError("explicit supported repository source policy required")
    if value != "legacy" and (purpose != "coding" or provider != "codex_cli"
            or source_view_policy != "legacy" or census_only):
        raise ValueError("repository source tools require a Codex coding route without the old source helper")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _path(value):
    if (type(value) is not str or not 1 <= len(value.encode("utf-8")) <= 4096
            or any(ord(char) < 32 for char in value)):
        raise ValueError("bounded canonical repository tool path required")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError("canonical absolute repository tool path required")
    return path


def _instruction(binding):
    representation = binding["config"]["representation"]
    return (
        "Repository source tools are configured for this invocation:\n"
        "Use repository_status and repository_list to discover the indexed checkout, "
        "repository_symbols for Python definitions, and repository_read for source reads. "
        f"The selected default representation is {representation}. "
        "Use repository_replace_symbol with the current source hash for supported edits; "
        "it changes the allocated canonical checkout and refreshes the repository view. "
        "Use repository_refresh after direct terminal edits; stale indexed reads or edits are refused. "
        "Request original source when compact content is insufficient. Terminal commands and tests "
        "still execute against the canonical checkout. Respect the task's existing allowed files. "
        "These tools do not prove behavior or establish completion. Tool availability, actual use, "
        "and token savings are measured separately.\n\n"
    )


def create_binding(*, workspace: Path, requested_policy: str):
    validate_policy(requested_policy, purpose="coding", provider="codex_cli")
    if requested_policy == "legacy":
        raise ValueError("legacy policy has no repository source binding")
    workspace = _path(str(workspace))
    helper = Path(__file__).with_name("repository_source_mcp.py").resolve(strict=True)
    source = _read_file(helper, 256_000)
    directory = Path("/tmp") / ("supervisor-repository-source-" + uuid.uuid4().hex)
    config = {"schema": "repository-source-mcp-config@1", "workspace": str(workspace),
              "state_dir": str(directory / "state"), "audit_store": str(directory / "operations.jsonl"),
              "representation": "original" if requested_policy == POLICIES[1] else "compact"}
    binding = {"schema": SCHEMA, "requested_policy": requested_policy,
        "python_executable": str(Path(sys.executable).absolute()), "helper_script": str(helper),
        "helper_sha256": _sha(source), "workspace": str(workspace), "invocation_dir": str(directory),
        "config_path": str(directory / "config.json"), "config_sha256": _sha(_json(config)),
        "config": config, "state_dir": config["state_dir"], "audit_store": config["audit_store"],
        **dict.fromkeys(_FALSE_FIELDS, False)}
    instruction = _instruction(binding)
    binding.update(instruction_sha256=_sha(instruction.encode()), instruction_bytes=len(instruction.encode()))
    validate_binding(binding, workspace=workspace)
    return binding


def _read_file(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValueError("bounded regular repository tool file required")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        if len(data) > limit:
            raise ValueError("repository tool file exceeds bound")
        return data
    finally:
        os.close(fd)


def validate_binding(binding, *, workspace=None, materialized=False):
    fields = {"schema", "requested_policy", "python_executable", "helper_script", "helper_sha256",
        "workspace", "invocation_dir", "config_path", "config_sha256", "config", "state_dir",
        "audit_store", "instruction_sha256", "instruction_bytes", *_FALSE_FIELDS}
    if (type(binding) is not dict or set(binding) not in (fields, fields | {"usage"})
            or binding.get("schema") != SCHEMA
            or type(binding.get("requested_policy")) is not str or binding["requested_policy"] not in POLICIES[1:]
            or any(binding.get(key) is not False for key in _FALSE_FIELDS)):
        raise ValueError("invalid repository source binding")
    paths = {key: _path(binding[key]) for key in ("python_executable", "helper_script", "workspace",
        "invocation_dir", "config_path", "state_dir", "audit_store")}
    directory = paths["invocation_dir"]
    helper = Path(__file__).with_name("repository_source_mcp.py").resolve(strict=True)
    if (paths["helper_script"].name != helper.name or paths["helper_script"].is_relative_to(paths["workspace"])
            or re.fullmatch(r"/tmp/supervisor-repository-source-[0-9a-f]{32}", str(directory)) is None
            or directory.is_relative_to(paths["workspace"])
            or (workspace is not None and paths["workspace"] != Path(workspace))
            or paths["config_path"] != directory / "config.json"
            or paths["state_dir"] != directory / "state"
            or paths["audit_store"] != directory / "operations.jsonl"):
        raise ValueError("repository source binding identity differs")
    config = {"schema": "repository-source-mcp-config@1", "workspace": binding["workspace"],
              "state_dir": binding["state_dir"], "audit_store": binding["audit_store"],
              "representation": "original" if binding["requested_policy"] == POLICIES[1] else "compact"}
    if type(binding["config"]) is not dict or binding["config"] != config:
        raise ValueError("repository server configuration differs")
    for key in ("helper_sha256", "config_sha256", "instruction_sha256"):
        if type(binding[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", binding[key]) is None:
            raise ValueError("repository source digest required")
    instruction = _instruction(binding)
    if (binding["config_sha256"] != _sha(_json(config))
            or binding["instruction_sha256"] != _sha(instruction.encode())
            or type(binding["instruction_bytes"]) is not int
            or binding["instruction_bytes"] != len(instruction.encode())):
        raise ValueError("repository source binding content differs")
    if materialized:
        if (paths["helper_script"] != helper
                or paths["python_executable"] != Path(sys.executable).absolute()):
            raise ValueError("repository tool runtime implementation differs")
        info = directory.lstat()
        if (not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700 or directory.resolve() != directory):
            raise ValueError("private repository tool directory required")
        if (_read_file(paths["config_path"], 16_384) != _json(config)
                or _sha(_read_file(helper, 256_000)) != binding["helper_sha256"]):
            raise ValueError("repository tool source or configuration changed")
    return instruction


def create_runtime_store(binding):
    validate_binding(binding)
    directory = Path(binding["invocation_dir"])
    directory.mkdir(mode=0o700)
    info = directory.lstat()
    identity = (info.st_dev, info.st_ino)
    try:
        Path(binding["state_dir"]).mkdir(mode=0o700)
        fd = os.open(binding["config_path"], os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "wb") as stream:
            stream.write(_json(binding["config"]))
            stream.flush()
            os.fsync(stream.fileno())
        validate_binding(binding, materialized=True)
        return identity
    except BaseException:
        finish_runtime_store(binding, expected_identity=identity)
        raise


def native_config_arguments(binding):
    """Return TOML argv values; never shell-interpolate configuration data."""
    validate_binding(binding, materialized=True)
    settings = {"command": binding["python_executable"], "args": [binding["helper_script"],
        "--config", binding["config_path"], "--config-sha256", binding["config_sha256"]],
        "env_vars": ["PYTHONPATH"], "startup_timeout_sec": 60, "tool_timeout_sec": 60,
        "enabled_tools": list(TOOLS), "required": True}
    table = "{" + ",".join(key + "=" + json.dumps(value, ensure_ascii=False)
                          for key, value in settings.items()) + "}"
    return ["-c", "mcp_servers.repository_source=" + table,
            "-c", "mcp_optional_startup_grace_ms=0"]


def _remove_contents(fd, *, remaining, depth=0):
    if depth > 64:
        raise ValueError("repository cleanup depth exceeded")
    os.fchmod(fd, 0o700)
    for name in os.listdir(fd):
        remaining[0] -= 1
        if remaining[0] < 0:
            raise ValueError("repository cleanup entry bound exceeded")
        info = os.stat(name, dir_fd=fd, follow_symlinks=False)
        if stat.S_ISDIR(info.st_mode):
            child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
            try:
                opened = os.fstat(child)
                if (opened.st_dev, opened.st_ino) != (info.st_dev, info.st_ino):
                    raise ValueError("repository cleanup directory changed")
                _remove_contents(child, remaining=remaining, depth=depth + 1)
            finally:
                os.close(child)
            current = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if (current.st_dev, current.st_ino) != (info.st_dev, info.st_ino):
                raise ValueError("repository cleanup directory changed")
            os.rmdir(name, dir_fd=fd)
        else:
            os.unlink(name, dir_fd=fd)


def finish_runtime_store(binding, *, expected_identity):
    usage = {"status": "unavailable", "initializations": None, "tool_calls": None,
        "successful_reads": None, "compact_reads": None, "canonical_edits": None, "refreshes": None,
        "errors": None, "uncertain_edits": None, "returned_bytes": None, "store_cleanup_complete": False,
        "artifacts_retained_after_invocation": None}
    if expected_identity is None:
        return usage
    directory = Path(binding["invocation_dir"])
    fd = None
    try:
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        info = os.fstat(fd)
        if ((info.st_dev, info.st_ino) != expected_identity or info.st_uid != os.getuid()
                or stat.S_IMODE(info.st_mode) != 0o700):
            raise ValueError("repository invocation directory changed")
        try:
            from .repository_source_mcp import summarize_repository_audit
            summary = summarize_repository_audit(Path(binding["audit_store"]))
            if (type(summary) is dict and summary.get("status") in {"observed", "not_observed", "unavailable"}
                    and all(value is None or type(value) is int and value >= 0
                            for key, value in summary.items() if key != "status")
                    and set(summary) == set(usage) - {"store_cleanup_complete", "artifacts_retained_after_invocation"}):
                usage.update(summary)
        except Exception:
            pass
        current = directory.lstat()
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ValueError("repository invocation directory changed")
        _remove_contents(fd, remaining=[32768])
        current = directory.lstat()
        if (current.st_dev, current.st_ino) != expected_identity:
            raise ValueError("repository invocation directory changed")
        directory.rmdir()
        usage.update(store_cleanup_complete=True, artifacts_retained_after_invocation=False)
    except Exception:
        usage["store_cleanup_error"] = "cleanup_unavailable"
    finally:
        if fd is not None:
            os.close(fd)
    return usage
