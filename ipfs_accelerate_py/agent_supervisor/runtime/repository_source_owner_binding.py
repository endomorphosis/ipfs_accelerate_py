"""Public worker binding for an independently owned, read-only repository MCP.

Creation and materialized validation check the live owner descriptor. Metadata
validation only reconstructs a historical prompt; it grants no live authority.
This module creates no store, lease, index, audit file, socket or subprocess.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys

from . import repository_source_owner_protocol as protocol

SCHEMA = "supervisor-repository-source-owner-binding@1"
POLICIES = ("repository-source-raw@1", "repository-source-compact@1")
_SOURCES = ("repository_source_owner_stdio.py", "repository_source_owner_protocol.py",
            "numerical_obligation_client.py")
_FORWARDING_SOURCE = "repository_source_worker_receipts.py"
_FALSE_FIELDS = ("native_tool_interception", "helper_adoption_guaranteed", "token_savings_measured",
                 "proof_authority", "execution_authority", "completion_authority")
_FIELDS = {"schema", "requested_policy", "workspace", "descriptor_path", "descriptor_sha256",
    "descriptor", "session_id", "representation", "owner_config_sha256", "owner_helper_sha256",
    "python_executable", "python_identity", "helper_script", "helper_sha256", "helper_sources",
    "config", "config_sha256", "instruction_sha256", "instruction_bytes", "read_only", *_FALSE_FIELDS}
_IDENTITY_FIELDS = {"resolved_path", "sha256", "device", "inode", "size", "mode", "uid", "gid",
                    "mtime_ns", "ctime_ns"}


def _need(condition, message="invalid repository owner binding"):
    if not condition:
        raise ValueError(message)


def _json(value):
    return protocol.canonical_json(value)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _digest(value):
    return type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _path(value):
    _need(type(value) is str and 1 <= len(value.encode("utf-8")) <= 4096
          and not any(ord(char) < 32 for char in value), "bounded canonical owner binding path required")
    path = Path(value)
    _need(path.is_absolute() and ".." not in path.parts and str(path) == value,
          "canonical absolute owner binding path required")
    return path


def _snapshot(info):
    return tuple(getattr(info, name) for name in ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid",
        "st_nlink", "st_size", "st_mtime_ns", "st_ctime_ns"))


def _file_identity(path, *, limit):
    """Hash a bounded regular file without following its final component."""
    before = path.lstat()
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        info = os.fstat(fd)
        _need(stat.S_ISREG(info.st_mode) and 0 < info.st_size <= limit,
              "bounded regular owner helper required")
        digest, total = hashlib.sha256(), 0
        while True:
            chunk = os.read(fd, min(65536, limit + 1 - total))
            if not chunk:
                break
            total += len(chunk)
            _need(total <= limit, "owner helper byte bound exceeded")
            digest.update(chunk)
        _need(total == info.st_size and _snapshot(before) == _snapshot(info)
              == _snapshot(os.fstat(fd)) == _snapshot(path.lstat()), "owner helper changed during read")
        return dict(resolved_path=str(path), sha256=digest.hexdigest(), device=info.st_dev,
            inode=info.st_ino, size=info.st_size, mode=info.st_mode, uid=info.st_uid, gid=info.st_gid,
            mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
    finally:
        os.close(fd)


def _helpers(*, forwarding=False):
    directory = Path(__file__).resolve(strict=True).parent
    return {name: directory / name for name in (*_SOURCES, *((_FORWARDING_SOURCE,) if forwarding else ()))}


def _source_hashes(*, forwarding=False):
    return {name: _file_identity(path, limit=256_000)["sha256"]
            for name, path in _helpers(forwarding=forwarding).items()}


def _python_identity():
    return _file_identity(Path(sys.executable).resolve(strict=True), limit=64 * 1024 * 1024)


def unavailable_usage():
    """Only owner audit records can establish actual use or cleanup."""
    return dict(status="unavailable", reason="owner_audit_required", initializations=None,
        tool_calls=None, successful_reads=None, compact_reads=None, canonical_edits=None,
        refreshes=None, errors=None, uncertain_edits=None, returned_bytes=None,
        store_cleanup_complete=None, artifacts_retained_after_invocation=None)


def _settings(binding):
    args = ["-I", "-B", binding["helper_script"],
        protocol.DESCRIPTOR_FLAG, binding["descriptor_path"], protocol.DIGEST_FLAG,
        binding["descriptor_sha256"], "--workspace", binding["workspace"]]
    if "worker_forwarding" in binding:
        args.extend(["--worker-forwarding-target", _json(binding["worker_forwarding"]).decode("utf-8")])
    return {"command": binding["python_executable"], "args": args,
        "env_vars": [], "startup_timeout_sec": 30, "tool_timeout_sec": 30,
        "enabled_tools": sorted(protocol.READ_ONLY_TOOL_NAMES), "required": True}


def _config(binding):
    return {"schema": "repository-source-owner-launch-config@1",
        **{key: binding[key] for key in ("workspace", "descriptor_path", "descriptor_sha256",
            "session_id", "representation", "owner_config_sha256", "owner_helper_sha256",
            "python_identity", "helper_sources")}, "settings": _settings(binding),
        **({"worker_forwarding": binding["worker_forwarding"]} if "worker_forwarding" in binding else {})}


def _instruction(binding):
    return (
        "Read-only repository source tools are configured for this invocation:\n"
        "Use repository_status and repository_list to discover the indexed checkout, "
        "repository_symbols for Python definitions, and repository_read for source reads. "
        "Use repository_context_query to select structural capsule references and "
        "repository_capsule_expand for selected details from the same generation and semantic root. "
        "Queries do not establish intent satisfaction. Read current source before editing from a capsule "
        "and respect its raw-source requirements. Definition reads are dedented and omit imports "
        "and enclosing context; omit symbol_id for a full file. "
        f"The selected default representation is {binding['representation']}. "
        "Request original source when compact content is insufficient or a symbol is unsupported. "
        "Make edits and run commands or tests against the allocated checkout, respecting the task's "
        "allowed files. Use repository_refresh after direct terminal edits; it refreshes only the "
        "owner's private index, and stale reads are refused. Repository tools cannot edit source. "
        "These tools do not prove behavior or establish completion. Tool availability, actual use, "
        "and token savings require separate evidence.\n\n"
    )


def create_binding(*, workspace, requested_policy, descriptor_path, descriptor_sha256, worker_forwarding=None):
    _need(type(requested_policy) is str and requested_policy in POLICIES)
    workspace, descriptor_path = _path(str(workspace)), _path(str(descriptor_path))
    _need(_digest(descriptor_sha256), "descriptor digest required")
    descriptor = protocol.load_descriptor(descriptor_path, descriptor_sha256, workspace=workspace)
    # Own a JSON copy: later mutations to the caller's descriptor cannot change
    # the recorded public session identity behind the launch configuration.
    descriptor = protocol.strict_json_loads(_json(descriptor))
    sources = _source_hashes()
    binding = dict(schema=SCHEMA, requested_policy=requested_policy, workspace=str(workspace),
        descriptor_path=str(descriptor_path), descriptor_sha256=descriptor_sha256,
        descriptor=descriptor, session_id=descriptor["session_id"],
        representation="original" if requested_policy == POLICIES[0] else "compact",
        owner_config_sha256=descriptor["config_sha256"], owner_helper_sha256=descriptor["helper_sha256"],
        python_executable=str(Path(sys.executable).absolute()), python_identity=_python_identity(),
        helper_script=str(_helpers()[_SOURCES[0]]), helper_sha256=sources[_SOURCES[0]],
        helper_sources=sources, read_only=True, **dict.fromkeys(_FALSE_FIELDS, False))
    binding["config"] = _config(binding)
    binding["config_sha256"] = _sha(_json(binding["config"]))
    instruction = _instruction(binding).encode("utf-8")
    binding.update(instruction_sha256=_sha(instruction), instruction_bytes=len(instruction))
    validate_binding(binding, workspace=workspace, materialized=True)
    return binding if worker_forwarding is None else with_worker_forwarding(binding, worker_forwarding)


def with_worker_forwarding(binding, target):
    """Bind advisory worker evidence to this launch without inspecting its files.

    The router owns setup and collection. A missing or replaced outbox cannot
    invalidate the live owner/MCP route or grant observation authority.
    """
    from .repository_source_worker_receipts import validate_target
    validate_binding(binding)
    target = validate_target(target, workspace=binding["workspace"], session_id=binding["session_id"],
                             descriptor_sha256=binding["descriptor_sha256"])
    _need(target["uid"] == binding["descriptor"]["worker_uid"], "worker forwarding UID differs")
    result = protocol.strict_json_loads(_json(binding))
    result["worker_forwarding"] = target
    result["helper_sources"] = _source_hashes(forwarding=True)
    result["helper_sha256"] = result["helper_sources"][_SOURCES[0]]
    result["config"] = _config(result)
    result["config_sha256"] = _sha(_json(result["config"]))
    validate_binding(result)
    return result


def helper_sources_match(binding):
    """Check current fixed helper bytes, without asserting owner liveness."""
    forwarding = type(binding) is dict and "worker_forwarding" in binding
    expected_sources = set(_SOURCES) | ({_FORWARDING_SOURCE} if forwarding else set())
    if (type(binding) is not dict or type(binding.get("helper_sources")) is not dict
            or set(binding["helper_sources"]) != expected_sources):
        return False
    try:
        return (binding.get("helper_sha256") == binding["helper_sources"][_SOURCES[0]]
                and binding["helper_sources"] == _source_hashes(forwarding=forwarding))
    except (OSError, ValueError):
        return False


def validate_binding(binding, *, workspace=None, materialized=False):
    """Reconstruct metadata, or additionally recheck the live worker boundary."""
    _need(type(materialized) is bool)
    _need(type(binding) is dict and set(binding) in (_FIELDS, _FIELDS | {"usage"},
          _FIELDS | {"worker_forwarding"}, _FIELDS | {"usage", "worker_forwarding"})
          and binding.get("schema") == SCHEMA and type(binding.get("requested_policy")) is str
          and binding["requested_policy"] in POLICIES and binding.get("read_only") is True
          and all(binding.get(key) is False for key in _FALSE_FIELDS))
    paths = {name: _path(binding[name]) for name in
             ("workspace", "descriptor_path", "python_executable", "helper_script")}
    _need(workspace is None or paths["workspace"] == _path(str(workspace)), "owner workspace differs")
    _need(paths["helper_script"].name == _SOURCES[0]
          and not paths["helper_script"].is_relative_to(paths["workspace"])
          and not paths["descriptor_path"].is_relative_to(paths["workspace"]),
          "owner implementation and descriptor must be outside the allocated checkout")
    for name in ("descriptor_sha256", "owner_config_sha256", "owner_helper_sha256", "helper_sha256",
                 "config_sha256", "instruction_sha256"):
        _need(_digest(binding[name]), "owner binding digest required")
    descriptor = protocol.validate_descriptor(binding["descriptor"])
    representation = "original" if binding["requested_policy"] == POLICIES[0] else "compact"
    _need(binding["descriptor_sha256"] == _sha(_json(descriptor))
          and descriptor["workspace"] == binding["workspace"]
          and descriptor["session_id"] == binding["session_id"]
          and descriptor["representation"] == binding["representation"] == representation
          and descriptor["config_sha256"] == binding["owner_config_sha256"]
          and descriptor["helper_sha256"] == binding["owner_helper_sha256"], "owner descriptor binding differs")
    sources = binding["helper_sources"]
    expected_sources = set(_SOURCES)
    if "worker_forwarding" in binding:
        from .repository_source_worker_receipts import validate_target
        target = validate_target(binding["worker_forwarding"], workspace=binding["workspace"],
            session_id=binding["session_id"], descriptor_sha256=binding["descriptor_sha256"])
        _need(target["uid"] == descriptor["worker_uid"], "worker forwarding UID differs")
        expected_sources.add(_FORWARDING_SOURCE)
    _need(type(sources) is dict and set(sources) == expected_sources
          and all(_digest(value) for value in sources.values())
          and sources[_SOURCES[0]] == binding["helper_sha256"], "fixed owner helper pins required")
    identity = binding["python_identity"]
    _need(type(identity) is dict and set(identity) == _IDENTITY_FIELDS
          and _digest(identity["sha256"]), "fixed Python identity required")
    _path(identity["resolved_path"])
    _need(all(type(identity[key]) is int and identity[key] >= 0
              for key in _IDENTITY_FIELDS - {"resolved_path", "sha256"})
          and 0 < identity["size"] <= 64 * 1024 * 1024 and identity["inode"] > 0
          and stat.S_ISREG(identity["mode"]), "invalid Python file identity")
    instruction = _instruction(binding)
    _need(type(binding["config"]) is dict and _json(binding["config"]) == _json(_config(binding))
          and binding["config_sha256"] == _sha(_json(binding["config"]))
          and binding["instruction_sha256"] == _sha(instruction.encode("utf-8"))
          and type(binding["instruction_bytes"]) is int
          and binding["instruction_bytes"] == len(instruction.encode("utf-8")),
          "owner binding content differs")
    if "usage" in binding:
        _need(_json(binding["usage"]) == _json(unavailable_usage()), "owner usage requires a separate owner audit")
    if materialized:
        _need(paths["python_executable"] == Path(sys.executable).absolute()
              and paths["helper_script"] == _helpers()[_SOURCES[0]]
              and _json(identity) == _json(_python_identity()) and helper_sources_match(binding),
              "owner bridge runtime implementation differs")
        live = protocol.load_descriptor(paths["descriptor_path"], binding["descriptor_sha256"],
                                        workspace=paths["workspace"])
        _need(_json(live) == _json(descriptor), "live owner descriptor differs")
    return instruction


def native_config_arguments(binding):
    """Fixed isolated stdio bridge, represented as argv rather than shell text."""
    validate_binding(binding, materialized=True)
    table = "{" + ",".join(key + "=" + json.dumps(value, ensure_ascii=False)
                            for key, value in _settings(binding).items()) + "}"
    return ["-c", "mcp_servers.repository_source=" + table,
            "-c", "mcp_optional_startup_grace_ms=0"]
