"""Read retained world-packet input under an independently selected private root.

Historical reconstruction does not restore a live packet or consult source,
intent, database, signature or provider owners. Artifact names come from the
captured invocation ID; no receipt chooses a directory or an arbitrary path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import stat


SCHEMA = "terminal-world-packet-input-audit@1"
MAX_RECEIPT_BYTES = 256 * 1024
MAX_RECEIPT_NODES = 100_000
MAX_DEPTH = 64
FALSE_FIELDS = frozenset({"source_currentness_verified", "owner_currentness_verified",
    "owner_signature_replayed", "provider_delivery_verified", "token_counts_replayed",
    "proof_authority", "execution_authority", "completion_authority", "live_packet_issued"})
_EXCLUDED = frozenset({"semantic_metadata_candidate", "representation_displacement",
    "source_view_retrieval", "terminal_semantic_scope_dispatch_rechecked"})


class TerminalWorldPacketAuditError(ValueError):
    """A bounded retained-input or independent-root check failed."""


def _need(condition, code):
    if not condition:
        raise TerminalWorldPacketAuditError(code)


def _bounded(value):
    """Check finite selected JSON before a detached owner projection is copied."""
    pending = [(value, 0)]
    count = size = 0
    while pending:
        item, depth = pending.pop()
        count += 1
        _need(count <= MAX_RECEIPT_NODES and depth <= MAX_DEPTH, "world_audit_receipt_structure_bound")
        if type(item) is dict:
            _need(len(item) * 2 + len(pending) <= MAX_RECEIPT_NODES - count
                and all(type(key) is str for key in item), "world_audit_receipt_structure_bound")
            size += 2 + max(0, len(item) - 1) + len(item)
            for key, child in item.items():
                pending.extend(((key, depth + 1), (child, depth + 1)))
        elif type(item) is list or type(item) is tuple:
            _need(len(item) + len(pending) <= MAX_RECEIPT_NODES - count, "world_audit_receipt_structure_bound")
            size += 2 + max(0, len(item) - 1)
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            _need(len(item) <= MAX_RECEIPT_BYTES, "world_audit_receipt_byte_bound")
            # Charge UTF-8 and JSON escaping before retaining the encoded copy.
            size += 2
            for char in item:
                point = ord(char)
                _need(not 0xD800 <= point <= 0xDFFF, "world_audit_receipt_unicode")
                size += (2 if char in '"\\\b\f\n\r\t' else 6 if point < 32 else
                    1 if point < 128 else 2 if point < 2048 else 3 if point < 65536 else 4)
                _need(size <= MAX_RECEIPT_BYTES, "world_audit_receipt_byte_bound")
        elif item is None:
            size += 4
        elif type(item) is bool:
            size += 4 if item else 5
        elif type(item) is int:
            _need(item.bit_length() <= 4096, "world_audit_receipt_integer_bound")
            size += len(str(item))
        elif type(item) is float:
            import math
            _need(math.isfinite(item), "world_audit_receipt_nonfinite")
            size += len(json.dumps(item, allow_nan=False))
        else:
            _need(False, "world_audit_receipt_non_json")
        _need(size <= MAX_RECEIPT_BYTES, "world_audit_receipt_byte_bound")


def validate_world_packet_audit_receipt(receipt, *, workspace_root):
    """Project captured input fields without treating them as fresh authority."""
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_invocation_replay as replay
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_artifact_retention import (
        validate_world_packet_artifacts,
    )

    _need(type(receipt) is dict and len(receipt) <= 256
        and receipt.get("schema") == "router-implementation-invocation@1"
        and receipt.get("phase") == receipt.get("purpose") == "coding"
        and receipt.get("provider") == "codex_cli" and receipt.get("completion_authority") is False,
        "world_audit_invocation_route")
    _need(all(key in receipt for key in replay._REQUIRED), "world_audit_invocation_fields")
    _need(not any(receipt.get(key) is not None for key in replay._INCOMPATIBLE | _EXCLUDED)
        and all(receipt.get(key) in (None, "legacy") for key in ("source_input_policy", "query_reference_policy")),
        "world_audit_incompatible_input")
    identity = receipt.get("invocation_id")
    _need(type(identity) is str and re.fullmatch(r"[0-9a-f]{32}", identity), "world_audit_invocation_id")
    location = receipt.get("workspace")
    _need(type(location) is str and 0 < len(location.encode()) <= 4096, "world_audit_workspace")
    workspace, root = Path(location), Path(workspace_root)
    _need(workspace.is_absolute() and str(workspace) == location and ".." not in workspace.parts
        and root.is_absolute() and ".." not in root.parts and workspace != root
        and workspace.is_relative_to(root), "world_audit_workspace_root")
    selected = {key: receipt[key] for key in replay._REQUIRED | replay._OPTIONAL if key in receipt}
    selected["world_packet_artifacts"] = receipt.get("world_packet_artifacts")
    _bounded(selected)
    artifacts = validate_world_packet_artifacts(selected.pop("world_packet_artifacts"), invocation_id=identity)
    projection = replay._projection(selected)
    for prefix in ("prompt", "native_prompt", "router_prompt", "model_prompt", "workspace_advisory"):
        _need(type(projection[prefix + "_sha256"]) is str
            and re.fullmatch(r"[0-9a-f]{64}", projection[prefix + "_sha256"])
            and type(projection[prefix + "_bytes"]) is int
            and 0 < projection[prefix + "_bytes"] <= 262144, "world_audit_input_identity")
    _need(type(projection["model"]) is str and projection["model"].strip()
        and len(projection["model"].encode()) <= 512, "world_audit_model")
    _need(type(projection["world_task_packet"]) is dict, "world_audit_packet_receipt")
    return {**projection, "phase": "coding", "world_packet_artifacts": artifacts}


def _root_identity(info):
    return info.st_dev, info.st_ino, info.st_uid, stat.S_IMODE(info.st_mode)


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode, info.st_nlink,
            info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _open_root(root, workspace):
    root = Path(root)
    _need(root.is_absolute() and ".." not in root.parts and root != Path(workspace)
        and not root.is_relative_to(Path(workspace)), "world_audit_artifact_root")
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in root.parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_fd
        info = os.fstat(descriptor)
        _need(stat.S_ISDIR(info.st_mode) and info.st_uid == os.geteuid()
            and stat.S_IMODE(info.st_mode) == 0o700 and root.resolve(strict=True) == root
            and _root_identity(root.stat(follow_symlinks=False)) == _root_identity(info),
            "world_audit_private_root_custody")
        return descriptor, info
    except BaseException:
        os.close(descriptor)
        raise


def _read(root_fd, declared, limit):
    descriptor = os.open(declared["name"], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=root_fd)
    try:
        before = os.fstat(descriptor)
        _need(stat.S_ISREG(before.st_mode) and before.st_uid == os.geteuid()
            and stat.S_IMODE(before.st_mode) == 0o600 and before.st_nlink == 1
            and before.st_size == declared["bytes"] and 0 < before.st_size <= limit,
            "world_audit_artifact_custody")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
        after = os.fstat(descriptor)
        current = os.stat(declared["name"], dir_fd=root_fd, follow_symlinks=False)
        _need(len(data) == declared["bytes"] and hashlib.sha256(data).hexdigest() == declared["sha256"]
            and _file_identity(before) == _file_identity(after) == _file_identity(current),
            "world_audit_artifact_pin_or_identity")
        return data, _file_identity(before)
    finally:
        os.close(descriptor)


@dataclass(frozen=True)
class AuditedWorldPacketInput:
    """Historical input text kept private; public output contains hashes/counts."""

    model_prompt: str = field(repr=False)
    workspace_advisory: str = field(repr=False)
    _receipt_json: str = field(repr=False)

    def receipt(self):
        return json.loads(self._receipt_json)


def world_packet_artifact_root(*, state, explicit_root, workspace):
    """Default only from a current independently supplied private state root."""
    if explicit_root is not None:
        return Path(explicit_root)
    root = Path(state)
    descriptor, before = _open_root(root, workspace)
    try:
        _need(_root_identity(os.fstat(descriptor)) == _root_identity(before)
            == _root_identity(root.stat(follow_symlinks=False)), "world_audit_state_root_changed")
    finally:
        os.close(descriptor)
    return root / "world_packet_artifacts"


def replay_retained_world_packet_input(receipt, *, artifact_root, workspace_root):
    """Join independently retained descriptors and original input projection."""
    from ipfs_accelerate_py.agent_supervisor.runtime import repository_world_invocation_replay as replay
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet import REPLAY_MAX_BYTES
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata

    captured = validate_world_packet_audit_receipt(receipt, workspace_root=workspace_root)
    binding = captured["world_packet_artifacts"]
    _need(binding["status"] == "retained", "world_audit_artifact_unavailable")
    root_fd, root_info = _open_root(artifact_root, captured["workspace"])
    try:
        packet_bytes, packet_identity = _read(root_fd, binding["packet"], REPLAY_MAX_BYTES)
        invocation_bytes, invocation_identity = _read(root_fd, binding["invocation"], replay.MAX_BYTES)
        artifact = replay._decode_canonical(invocation_bytes, binding["invocation"]["sha256"], max_bytes=replay.MAX_BYTES)
        _need(artifact.get("packet_artifact_sha256") == binding["packet"]["sha256"]
            and replay._canonical_bytes(artifact.get("packet_artifact")) == packet_bytes,
            "world_audit_packet_invocation_join")
        _need(replay._canonical_bytes(artifact.get("invocation")) == replay._canonical_bytes(replay._projection(captured)),
            "world_audit_original_invocation_join")
        reconstructed = replay.replay_repository_world_invocation(invocation_bytes,
            expected_sha256=binding["invocation"]["sha256"])
        with public_replay_without_metadata():
            _model, advisory = render_model_prompt(prompt="", purpose="coding",
                workspace=Path(captured["workspace"]), world_packet=True,
                tool_output_binding=captured.get("tool_output_policy"),
                repository_source_binding=captured.get("repository_source_policy"))
        _need(hashlib.sha256(advisory.encode()).hexdigest() == captured["workspace_advisory_sha256"]
            and len(advisory.encode()) == captured["workspace_advisory_bytes"], "world_audit_advisory_join")
        root = Path(artifact_root)
        _need(root.resolve(strict=True) == root and _root_identity(os.fstat(root_fd)) == _root_identity(root_info)
            == _root_identity(root.stat(follow_symlinks=False)), "world_audit_root_changed")
        _need(all(_file_identity(os.stat(binding[kind]["name"], dir_fd=root_fd, follow_symlinks=False)) == identity
            for kind, identity in (("packet", packet_identity), ("invocation", invocation_identity))),
            "world_audit_artifact_changed_after_replay")
        diagnostic = dict(schema=SCHEMA, verification_scope="historical_world_packet_input_reconstruction",
            invocation_id=captured["invocation_id"], packet_artifact_sha256=binding["packet"]["sha256"],
            invocation_artifact_sha256=binding["invocation"]["sha256"],
            model_input_verified=True, historical_input_reconstructed=True,
            recorded_provider_attempted=captured["world_task_packet"]["provider_attempted"],
            replay=reconstructed.receipt(), **dict.fromkeys(FALSE_FIELDS, False))
        return AuditedWorldPacketInput(reconstructed.model_prompt, advisory,
            replay._canonical_bytes(diagnostic).decode())
    except TerminalWorldPacketAuditError:
        raise
    except Exception as error:
        raise TerminalWorldPacketAuditError("world_audit_reconstruction_failed") from error
    finally:
        os.close(root_fd)


__all__ = ["SCHEMA", "FALSE_FIELDS", "TerminalWorldPacketAuditError", "AuditedWorldPacketInput",
    "validate_world_packet_audit_receipt", "replay_retained_world_packet_input", "world_packet_artifact_root"]
