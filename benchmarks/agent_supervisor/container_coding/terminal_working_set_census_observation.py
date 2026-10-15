"""Owner replay of a closed census and its stopped source-reader session.

Reconstruct both complete inputs from current owner artifacts. The worker's
counts and hashes are accepted only if the existing selector/exporter reproduces
the entire observation. No provider is initialized and no prompt is exported.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .terminal_live_metadata_census import _read_metadata
from .terminal_numerical_retrieval_observation import _AUDIT_FIELDS, _records


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _need(condition):
    if not condition:
        raise ValueError("working-set census differs from current owner replay")


def replay_working_set_census(*, prepared, observed, deployment, provider_selection,
        source_view_profile, source_view_policy, source_audit_paths):
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_client as source
    from ipfs_accelerate_py.agent_supervisor.runtime import worker_source_request_observation as operations
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_router_translation as codec
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_working_set as working
    from ipfs_accelerate_py.agent_supervisor.runtime.semantic_working_set_census import working_set_census_receipt
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_residual_context import load_doctor_residual_advisory
    from ipfs_accelerate_py.agent_supervisor.runtime.router_public_instruction import load_public_instruction
    from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import (
        validate_source_formalization_context, _protected_directory,
    )
    from .terminal_live_census_provenance import _worker_boundary
    from .terminal_source_view_policy import source_view_observation

    options = prepared["runner_kwargs"]
    repository, workspace = prepared["repository"], prepared["workspace"]
    # This validates the independently qualified deployment and allocated scope;
    # it is not a fabricated planning/provider invocation or a usage receipt.
    boundary = _worker_boundary(deployment, dict(workspace=str(workspace),
        external_container_boundary=deployment["qualification"]["permission_evidence"]["boundary"]),
        provider_selection["provider"])
    _need(type(source_audit_paths) is list and len(source_audit_paths) == 1)
    audit_path = Path(source_audit_paths[0])
    _protected_directory(audit_path.parent, owner=os.geteuid(), group=os.getegid())
    audit = _read_metadata(audit_path)
    _need(set(audit) == _AUDIT_FIELDS and audit["schema"] == operations.EVIDENCE_AUDIT_SCHEMA)
    _need(all(audit[key] is False for key in ("proof_authority", "execution_authority", "completion_authority")))
    _need(all(type(audit[key]) is int and audit[key] == 0 for key in
        ("retrieval_provider_calls", "source_obligations_discharged")))
    descriptor = source.validate_descriptor(audit["session"])
    _need(descriptor["schema"] == "supervisor-worker-source-descriptor@4"
        and descriptor["semantic_evidence_mode"] == "selected-capsules@1"
        and descriptor["source_edit_mode"] == "disabled"
        and descriptor["workspace"] == str(workspace)
        and descriptor["owner_uid"] == os.geteuid() == boundary["owner_uid"]
        and descriptor["worker_uid"] == boundary["worker_uid"]
        and descriptor["boundary_manifest_sha256"] == boundary["manifest_sha256"]
        and audit["descriptor_sha256"] == _sha(source.canonical_json(descriptor))
        and audit_path.name == "source-view-" + descriptor["session_id"] + "-audit.json")
    _need(not Path(descriptor["socket_path"]).exists()
        and not (audit_path.parent / ("source-view-" + descriptor["session_id"] + ".json")).exists())
    expected_policy = source_view_observation(source_view_profile, source_view_policy)
    _need(descriptor["view_policy_binding_sha256"] == expected_policy.get("source_view_policy_binding_sha256"))

    prompt = prepared["prompt_artifact"].read_text()
    _need(_sha(prompt.encode()) == prepared["metadata"]["native_prompt_sha256"])
    encoded = codec.encode_semantic_router_prompt(prompt=prompt, repository=repository)
    binding = source.semantic_binding(encoded, source_view_profile)
    _need(all(descriptor[key] == value for key, value in binding.items()))
    view = working.project_semantic_working_set(encoded, descriptor)
    guidance = source.guidance(source_view_profile)
    residual_advice, residual_receipt = "", None
    if options.get("doctor_residual_artifact") is not None:
        residual_advice, residual_receipt = load_doctor_residual_advisory(
            artifact=options["doctor_residual_artifact"], expected_sha256=options["doctor_residual_sha256"],
            repository=repository, task_cid=options["doctor_residual_task_cid"], prompt=prompt, workspace=workspace)
    instruction, instruction_receipt = load_public_instruction(
        artifact=options["public_instruction_artifact"], expected_sha256=options["public_instruction_sha256"],
        task_cid=options["public_instruction_task_cid"], prompt=prompt, workspace=workspace, repository=repository)

    def render(transport):
        model_prompt, advisory = render_model_prompt(prompt=transport + guidance + residual_advice + instruction,
            purpose="coding", workspace=workspace, semantic_transport=True)
        model_prompt, reply = apply_coding_reply_contract(model_prompt=model_prompt,
            mode=options["coding_reply_mode"], purpose="coding", provider=provider_selection["provider"])
        return model_prompt, advisory, reply

    native, advisory, native_reply = render(encoded.provider_prompt)
    candidate, candidate_advisory, candidate_reply = render(view.provider_prompt)
    _need(advisory == candidate_advisory)
    selection = working.select_working_set(view, native_complete_prompt=native, candidate_complete_prompt=candidate)
    selected = selection.selected_mode != "legacy"
    _need(type(audit["requests"]) is list and len(audit["requests"]) == int(selected))
    records, delivered_evidence_bytes = [], 0
    for item in audit["requests"]:
        _need(type(item) is dict)
        operations.validate_record(item)
        record = dict(item)
        for field in operations.FIELDS:
            record.pop(field)
        _need(record.pop("semantic_evidence_page", None) is True)
        size = record.pop("returned_semantic_evidence_bytes", None)
        _need(type(size) is int and 0 < size <= record["response_bytes"] <= 4096)
        _need(record.pop("candidate_context_page", None) is False
            and record.pop("discovery_page", None) is False)
        for key in ("returned_candidate_context_bytes", "returned_discovery_rows"):
            value = record.pop(key, None)
            _need(type(value) is int and value == 0)
        _need(record["status"] == "ok" and record["delivered"] is True
            and type(record["returned_source_bytes"]) is int and record["returned_source_bytes"] == 0)
        delivered_evidence_bytes += size
        records.append(record)
    counts, _, _, _ = _records(dict(audit, requests=records), descriptor,
        detail_field="returned_source_bytes", count_noun="source_bytes")
    preflight = None
    if selected:
        import json
        reference = json.loads(view.provider_prompt)["working_set"]["capsule_references"][0]
        row = audit["requests"][0]
        preflight = dict(schema="supervisor-semantic-working-set-preflight@1", requests=1,
            session_id=descriptor["session_id"], capsule_index=reference["capsule_index"],
            capsule_sha256=reference["capsule_sha256"], response_bytes=row["response_bytes"],
            response_sha256=row["response_sha256"], provider_input_bytes=0,
            source_freshness_checked=True, proof_authority=False)
    source_receipt = dict(schema="worker-source-dispatch-observation@4",
        descriptor_sha256=audit["descriptor_sha256"], session_id=descriptor["session_id"], **binding,
        boundary_manifest_sha256=boundary["manifest_sha256"], guidance_sha256=_sha(guidance.encode()),
        guidance_bytes=len(guidance.encode()), requests_observed=False,
        proof_authority=False, execution_authority=False, completion_authority=False,
        **{key: descriptor[key] for key in ("source_edit_mode", "view_policy_binding_sha256",
            "source_scope_sha256", "semantic_evidence_mode")})
    formalization = validate_source_formalization_context(prompt=prompt, repository=repository, workspace=workspace)
    if formalization is not None:
        # Rebuild the worker's observation after independent owner-state replay.
        # These two flags describe its side of the handoff, not this replay.
        formalization = dict(formalization, owner_validation_inherited=True, owner_private_state_replayed_here=False)
    expected = working_set_census_receipt(selection=selection, encoded=encoded, prompt=prompt,
        model=provider_selection["model"], workspace=workspace, provider=provider_selection["provider"],
        components=codec.census_semantic_router_prompt(encoded=encoded, census_schema=codec.DETAILED_COMPONENT_CENSUS_SCHEMA),
        source_receipt=source_receipt, preflight=preflight, advisory=advisory,
        boundary_sha256=boundary["manifest_sha256"], reply=candidate_reply if selected else native_reply,
        formalization=formalization, doctor=residual_receipt, public=instruction_receipt)
    _need(source.canonical_json(expected) == source.canonical_json(observed))
    return dict(schema="terminal-working-set-census-source-observation@1", observation_only=True,
        descriptor_sha256=audit["descriptor_sha256"], audit_sha256=_sha(source.canonical_json(audit)),
        deployment_sha256=_sha(source.canonical_json(deployment)),
        complete_input_replayed=True, closed_observation_replayed=True,
        source_session_stopped=True, source_session_accounting_complete=True,
        source_session_requests=counts["accepted_requests"],
        source_response_bytes=counts["delivered_response_bytes"],
        source_evidence_bytes=delivered_evidence_bytes, provider_input_bytes=0,
        provider_calls=0, provider_tokens=None, total_token_savings_measured=False,
        proof_authority=False, completion_authority=False, raw_content_exported=False)
