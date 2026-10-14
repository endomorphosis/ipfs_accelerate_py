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
        from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_mcp
        source = Path(repository_source_mcp.__file__).read_bytes()
        checks["repository_source_helper_source"] = (
            hashlib.sha256(source).hexdigest() == receipt["repository_source_policy"]["helper_sha256"])
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
    return result
