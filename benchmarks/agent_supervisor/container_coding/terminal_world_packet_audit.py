"""Join pinned historical world input to retained Terminal context meaning.

The independent router receipt supplies the artifact pin. Reconstruction checks
recorded input bytes only; it does not attest delivery, freshness, signatures,
provider token accounting, or task success. A selected ordinary nomination
bundle additionally requires its retained preparation repository and exact
bundle bytes, which must select no formalization contexts for this route.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import re


REFERENCE_SCHEMA = "repository-world-invocation-artifact-reference@1"
MAX_ARTIFACT_BYTES = 3 * 1024 * 1024
MAX_AGGREGATE_ARTIFACT_BYTES = 12 * 1024 * 1024
_REFERENCE_FIELDS = {"schema", "invocation_id", "artifact_sha256", "artifact_bytes",
                     "packet_artifact_sha256"}
_CONTEXT_SCHEMAS = {
    "semantic-context": {"supervisor-semantic-worker-context@1", "supervisor-semantic-worker-context@2"},
    "code-retrieval-context": {"supervisor-code-retrieval-context@1"},
    "intent-world-context": {"supervisor-intent-world-planning-context@1"},
}


def validate_world_artifact_reference(value, *, invocation_id):
    """Validate an independently collected reference, never a self-pin file."""
    if (type(value) is not dict or set(value) != _REFERENCE_FIELDS
            or value["schema"] != REFERENCE_SCHEMA
            or type(invocation_id) is not str
            or re.fullmatch(r"[0-9a-f]{32}", invocation_id) is None
            or value["invocation_id"] != invocation_id
            or type(value["artifact_bytes"]) is not int
            or not 0 < value["artifact_bytes"] <= MAX_ARTIFACT_BYTES
            or any(type(value[key]) is not str or re.fullmatch(r"[0-9a-f]{64}", value[key]) is None
                   for key in ("artifact_sha256", "packet_artifact_sha256"))):
        raise ValueError("invalid world invocation artifact reference")
    return dict(value)


def project_world_receipt(item, *, workspace_root=None):
    """Check the full receipt before selecting its bounded replay projection."""
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_invocation_replay import _projection

    if (type(item) is not dict or "world_task_packet" not in item
            or item.get("phase") != "coding" or item.get("purpose") != "coding"):
        raise ValueError("world replay requires a coding receipt")
    reference = validate_world_artifact_reference(item.get("world_packet_replay_artifact"),
        invocation_id=item.get("invocation_id"))
    # Incompatible transformations must be rejected before any fields disappear.
    projected = _projection(item)
    location = projected["workspace"]
    if type(location) is not str or "\x00" in location or len(location.encode()) > 4096:
        raise ValueError("invalid world workspace identity")
    workspace = Path(location)
    if not workspace.is_absolute() or ".." in workspace.parts or str(workspace) != location:
        raise ValueError("invalid world workspace identity")
    if workspace_root is not None:
        root = Path(workspace_root)
        if (not root.is_absolute() or ".." in root.parts or workspace == root
                or not workspace.is_relative_to(root)):
            raise ValueError("world workspace outside configured allocation root")
    return {**projected, "phase": "coding", "world_packet_replay_artifact": reference}


def _supported_preparation(expected, prepared, packet):
    """Do not reinterpret advanced selected evidence as ordinary context."""
    from .terminal_task_profile import SCOPED_SCHEMA

    profile = prepared.get("task_profile")
    if (any(key in expected for key in ("source384_context",
            "numerical_formalization", "numerical_selection_384", "terminal_semantic_scope"))
            or type(profile) is dict and profile.get("schema") == SCOPED_SCHEMA):
        raise ValueError("world replay unsupported retained formalization or signed scope")
    if "context_bundle" in expected:
        binding = expected["context_bundle"]
        if type(binding) is not dict or set(binding) != {"artifact", "sha256"}:
            raise ValueError("world replay requires an exact retained context bundle binding")
        # Ordinary Terminal preparation also publishes a nomination bundle.
        # _capsule's historical reader below must verify its exact retained
        # bytes, task identity and empty formalization selection. Presence alone
        # is neither a formalization selection nor permission to ignore it.
    evidence = packet["capsule"]["evidence"]
    if any(row.get("kind") == "source-formalization-context" for row in evidence):
        raise ValueError("world replay unsupported retained formalization scope")
    from .terminal_context_audit import _json

    # The legacy capsule checks predate explicit schema versions. Retain their
    # schema-less fixture compatibility, but never interpret a declared future
    # contract using the current fields. The full chunk/meaning checks follow.
    for kind, supported in _CONTEXT_SCHEMAS.items():
        references = sorted((row for row in evidence if row.get("kind") == kind),
                            key=lambda row: row["reference_id"])
        if not references:
            continue  # The required-context check below rejects this absence.
        payload = _json("".join(row["summary"] for row in references).encode("utf-8"))
        if type(payload) is not dict:
            raise ValueError("world replay context requires an object")
        if "schema" in payload and (type(payload["schema"]) is not str
                                     or payload["schema"] not in supported):
            raise ValueError("world replay unsupported explicit context schema")
        if kind == "code-retrieval-context" and "retrieval_schema" in payload:
            if (type(payload["retrieval_schema"]) is not str
                    or payload["retrieval_schema"] != "supervisor-empty-code-retrieval@1"):
                raise ValueError("world replay unsupported explicit retrieval schema")


def _helper_checks(receipt):
    checks = {}
    if receipt.get("repository_source_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime.repository_source_policy import helper_sources_match
        checks["repository_source_helper_source"] = helper_sources_match(receipt["repository_source_policy"])
    if receipt.get("tool_output_policy") is not None:
        from ipfs_accelerate_py.agent_supervisor.runtime import tool_output_artifact
        source = Path(tool_output_artifact.__file__).read_bytes()
        checks["tool_output_helper_source"] = (
            hashlib.sha256(source).hexdigest() == receipt["tool_output_policy"]["helper_sha256"])
    return checks


def audit_world_terminal_context(*, state: Path, receipts: list[dict], workspace_root: Path):
    """Audit one bounded population without reopening source or intent owners."""
    from . import terminal_context_audit as audit
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_invocation_replay import (
        _projection, replay_repository_world_invocation,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_packet_replay import (
        _canonical_bytes, _decode_canonical,
    )

    result = audit._base()
    result.update(coding_receipts_seen=len(receipts), usable_coding_receipts=0,
                  world_packet_replays=[])
    if len(receipts) > audit.MAX_RECEIPTS:
        raise ValueError("world receipt inventory exceeds audit bound")
    parsed, seen, aggregate = [], set(), 0
    for item in receipts:
        try:
            row = project_world_receipt(item, workspace_root=workspace_root)
            if row["invocation_id"] in seen:
                raise ValueError("duplicate world invocation")
            seen.add(row["invocation_id"])
            aggregate += row["world_packet_replay_artifact"]["artifact_bytes"]
            if aggregate > MAX_AGGREGATE_ARTIFACT_BYTES:
                raise ValueError("world replay artifact population exceeds byte bound")
            parsed.append(row)
        except (ValueError, KeyError, TypeError):
            result["read_errors"].append({"kind": "world_receipt", "error_type": "InvalidReceipt"})
    result["usable_coding_receipts"] = len(parsed)
    if not parsed:
        result["reason"] = "no_usable_coding_receipt"
        return result
    result["invocations"] = [{"invocation_id": row["invocation_id"],
        "native_input_verified": False, "model_input_verified": False,
        "status": "unverified", "reason": "world_packet_replay_unavailable"} for row in parsed]
    root_fd = audit._open_directory(state)
    try:
        expected_raw = audit._read_at(root_fd, "context-result.json")
        prepared_raw = audit._read_at(root_fd, "prepared.json")
        expected, prepared = audit._json(expected_raw), audit._json(prepared_raw)
        result.update(context_result_sha256=audit._sha(expected_raw),
                      prepared_state_sha256=audit._sha(prepared_raw))
        for row, invocation in zip(parsed, result["invocations"]):
            try:
                reference = row["world_packet_replay_artifact"]
                raw = audit._read_at(root_fd, "world-invocations/" + row["invocation_id"] + ".json",
                                     limit=reference["artifact_bytes"])
                if len(raw) != reference["artifact_bytes"]:
                    raise ValueError("world artifact byte count mismatch")
                replay = replay_repository_world_invocation(raw,
                    expected_sha256=reference["artifact_sha256"])
                diagnostic = replay.receipt()
                projection_sha = audit._sha(_canonical_bytes(_projection(row)))
                if (diagnostic["invocation_projection_sha256"] != projection_sha
                        or diagnostic["packet_artifact_sha256"] != reference["packet_artifact_sha256"]):
                    raise ValueError("world artifact differs from independently collected receipt")
                artifact = _decode_canonical(raw, reference["artifact_sha256"], max_bytes=MAX_ARTIFACT_BYTES)
                packet, recorded = artifact["packet_artifact"], artifact["invocation"]
                _supported_preparation(expected, prepared, packet)
                if (packet["recipe"]["task_id"] != prepared["spec"]["task_key"]
                        or packet["recipe"]["task_text"] != prepared["query"]):
                    raise ValueError("world task differs from retained public task")
                from .terminal_world_nomination_audit import audit_world_nominations
                nominations = audit_world_nominations(packet=packet, expected=expected, prepared=prepared)
                observed, rendered, formalization = audit._capsule(
                    _canonical_bytes(packet["capsule"]), expected=expected, prepared=prepared)
                if formalization is not None or observed.get("terminal_semantic_scope") is not None:
                    raise ValueError("world replay unsupported retained formalization or signed scope")
                observed["compiled_capsule_prompt_sha256"] = observed.pop("native_prompt_sha256")
                observed["compiled_capsule_prompt_bytes"] = observed.pop("native_prompt_bytes")
                observed["input_representation"] = "world_task_packet"
                result["capsules"].append(observed)
                if not observed["all_context_checks_passed"]:
                    raise ValueError("world capsule context meaning mismatch")
                checks = _helper_checks(row)
                if nominations is not None:
                    checks["historical_context_nomination_projections"] = True
                checks.update(world_invocation_projection=True, world_packet_builder=True,
                    world_public_task=True, compiled_required_context_preserved=True,
                    compiled_context_meaning=True,
                    model_prompt_sha256=audit._sha(replay.model_prompt.encode()) == row["model_prompt_sha256"],
                    model_prompt_bytes=len(replay.model_prompt.encode()) == row["model_prompt_bytes"],
                    workspace_advisory_sha256=recorded["workspace_advisory_sha256"] == row["workspace_advisory_sha256"],
                    workspace_advisory_bytes=recorded["workspace_advisory_bytes"] == row["workspace_advisory_bytes"])
                if "context_bundle" in expected:
                    # _capsule has joined this independently pinned nomination
                    # through read_task_context_historical_selection. This
                    # requires the retained repository/bundle, not the worker
                    # checkout, and establishes no current context validity.
                    checks["historical_context_bundle_empty_formalization_selection"] = True
                match = {"capsule_id": observed["capsule_id"], "invocation_id": row["invocation_id"],
                    "input_representation": "world_task_packet", "workspace_sha256": audit._sha(row["workspace"].encode()),
                    "native_prompt_sha256": row["native_prompt_sha256"], "native_prompt_bytes": row["native_prompt_bytes"],
                    "model_prompt_sha256": row["model_prompt_sha256"], "model_prompt_bytes": row["model_prompt_bytes"],
                    "workspace_advisory_sha256": row["workspace_advisory_sha256"],
                    "workspace_advisory_bytes": row["workspace_advisory_bytes"],
                    "compiled_capsule_prompt_sha256": observed["compiled_capsule_prompt_sha256"],
                    "compiled_capsule_prompt_bytes": observed["compiled_capsule_prompt_bytes"],
                    "native_task_cid": observed["task_cid"],
                    "world_packet_task_cid": row["world_task_packet"]["task_cid"],
                    "model_input_checks": checks, "model_input_verified": all(checks.values()),
                    "world_packet_replay": diagnostic}
                result["matches"].append(match)
                if nominations is not None:
                    match["world_context_nominations"] = nominations
                result["world_packet_replays"].append(diagnostic)
                invocation.update(native_input_verified=True, model_input_verified=match["model_input_verified"],
                    status="verified_exact_model_input" if match["model_input_verified"] else "verified_native_input_only",
                    reason="exact_world_context_and_input_digests_match" if match["model_input_verified"]
                        else "world_model_projection_mismatch")
                if match["model_input_verified"]:
                    # The final message, not the separately rendered compiled
                    # capsule, owns this census. Slice by UTF-8 bytes and pin
                    # the exact advisory before parsing the world preamble.
                    try:
                        model_bytes = replay.model_prompt.encode("utf-8")
                        size = row["workspace_advisory_bytes"]
                        if type(size) is not int or not 0 <= size <= min(8192, len(model_bytes)):
                            raise ValueError("world census advisory bound")
                        advisory_bytes = model_bytes[:size]
                        if audit._sha(advisory_bytes) != row["workspace_advisory_sha256"]:
                            raise ValueError("world census advisory pin")
                        advisory = advisory_bytes.decode("utf-8", errors="strict")
                    except Exception:
                        result["rendered_input_census"]["unavailable"].append({
                            "invocation_id_sha256": audit._sha(row["invocation_id"].encode()),
                            "reason": "census_bound_or_input_unavailable"})
                    else:
                        audit._add_rendered_input_census(result, model_prompt=replay.model_prompt,
                            workspace_advisory=advisory, invocation_id=row["invocation_id"],
                            input_representation="world_task_packet")
            except Exception as error:
                result["read_errors"].append({"kind": "world_packet_replay", "error_type": type(error).__name__})
                if isinstance(error, ValueError) and str(error).startswith("world replay unsupported retained"):
                    invocation["reason"] = "world_packet_unsupported_retained_formalization_or_signed_scope"
    finally:
        os.close(root_fd)
    result["any_native_input_verified"] = any(row["native_input_verified"] for row in result["invocations"])
    result["any_model_input_verified"] = any(row["model_input_verified"] for row in result["invocations"])
    result["all_observed_coding_inputs_verified"] = (len(parsed) == len(receipts)
        and all(row["model_input_verified"] for row in result["invocations"]))
    result["status"] = ("verified_all_observed_model_inputs" if result["all_observed_coding_inputs_verified"]
        else "verified_some_observed_model_inputs" if result["any_model_input_verified"]
        else "verified_native_input_only" if result["any_native_input_verified"] else "unverified")
    result["reason"] = "per_invocation_context_and_input_comparison"
    return audit._finish_rendered_input_census(result)


# Independently retained artifact descriptors use a separate historical route.
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

__all__ += ["validate_world_artifact_reference", "project_world_receipt", "audit_world_terminal_context"]
