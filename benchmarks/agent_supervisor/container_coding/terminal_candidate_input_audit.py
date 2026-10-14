"""Historical candidate input reconstruction from owner-private retained bytes.

This reader never reconstructs an admission handle, calls a provider, checks
current candidate sources, or promotes pending formal artifacts to proofs.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _need(condition):
    if not condition:
        raise ValueError("candidate historical input binding differs")


def validate_receipt(receipt):
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_dispatch_context import FLAGS, OBSERVATION_SCHEMA
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_reply_contract import apply_candidate_reply_contract
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import _candidate_round_guidance
    from .terminal_context_audit import _digest, _same_json

    dispatch = receipt["candidate_round_dispatch"]
    _need(type(dispatch) is dict and dispatch.get("schema") == OBSERVATION_SCHEMA
        and all(type(dispatch.get(k)) is type(v) and dispatch[k] == v for k, v in FLAGS.items())
        and dispatch.get("validation_phase") == "before_provider"
        and all(dispatch.get(k) is True for k in ("owner_validation_inherited",
            "producer_validation_inherited", "canonical_source_current", "workspace_source_current"))
        and all(dispatch.get(k) is False for k in ("neural_inference_replayed", "owner_private_state_replayed_here"))
        and type(dispatch.get("round_number")) is int and dispatch["round_number"] == 2)
    for key in ("session_id", "prior_session_id"):
        _need(type(dispatch.get(key)) is str and re.fullmatch(r"[0-9a-f]{32}", dispatch[key]) is not None)
    _need(dispatch["session_id"] != dispatch["prior_session_id"])
    for key in ("native_prompt_sha256", "descriptor_sha256", "prior_descriptor_sha256",
                "handoff_sha256", "workspace_generation", "round_context_sha256", "producer_binding_sha256"):
        _digest(dispatch[key])
    for key in ("semantic_translation", "public_instruction", "doctor_residual_context"):
        _need(receipt.get(key) is None)
    _need(not set(receipt).intersection({"source_view_retrieval", "source_formalization_context",
        "coding_reply_contract", "semantic_metadata_view", "semantic_metadata_native_coding_reply_contract",
        "representation_displacement", "semantic_working_set", "semantic_working_set_preflight"}))
    _need(dispatch["provider_input_sha256"] == receipt["prompt_sha256"]
        and dispatch["provider_input_bytes"] == receipt["prompt_bytes"]
        and dispatch["native_prompt_sha256"] == receipt["native_prompt_sha256"]
        and dispatch["native_prompt_bytes"] == receipt["native_prompt_bytes"])
    advice = _candidate_round_guidance(dispatch).encode()
    _need(_same_json(receipt["candidate_retrieval_guidance"], dict(sha256=_sha(advice), bytes=len(advice),
        requests_observed=False, proof_authority=False, completion_authority=False)))
    reply = receipt["candidate_reply_contract"]
    _, expected = apply_candidate_reply_contract(model_prompt="", binding=dispatch)
    dynamic = {"model_prompt_before_sha256", "model_prompt_before_bytes",
        "model_prompt_after_sha256", "model_prompt_after_bytes", "response_validated"}
    _need(type(reply) is dict and set(reply) == set(expected) and type(reply["response_validated"]) is bool
        and all(_same_json(reply[key], expected[key]) for key in expected if key not in dynamic))
    _need(type(receipt["router_prompt_bytes"]) is int
        and receipt["router_prompt_bytes"] == receipt["prompt_bytes"] + len(advice)
        and reply["model_prompt_before_bytes"] == receipt["router_prompt_bytes"] + receipt["workspace_advisory_bytes"]
        and reply["model_prompt_after_bytes"] == receipt["model_prompt_bytes"]
        and reply["model_prompt_after_sha256"] == receipt["model_prompt_sha256"]
        and reply["model_prompt_before_bytes"] + reply["instruction_bytes"] == receipt["model_prompt_bytes"])
    for key in ("model_prompt_before_sha256", "model_prompt_after_sha256"):
        _digest(reply[key])
    _digest(receipt["router_prompt_sha256"])
    # A preflight can be absent on a recorded pre-provider failure. Its absence
    # does not prevent exact byte reconstruction or establish a provider call.
    if "candidate_round_preflight" in receipt:
        probe = receipt["candidate_round_preflight"]
        _need(type(probe) is dict and probe.get("schema") == "supervisor-candidate-round-preflight@1"
            and type(probe.get("requests")) is int and probe["requests"] == 1
            and probe.get("session_id") == dispatch["session_id"]
            and probe.get("workspace_generation") == dispatch["workspace_generation"]
            and probe.get("artifact") == "summary"
            and probe.get("artifact_sha256") == dispatch["candidate_artifacts"]["summary"]["sha256"]
            and type(probe.get("response_bytes")) is int and 0 < probe["response_bytes"] <= 8192
            and type(probe.get("provider_input_bytes")) is int and probe["provider_input_bytes"] == 0
            and all(probe.get(k) is True for k in ("child_custody_checked", "source_freshness_checked", "producer_freshness_checked"))
            and probe.get("proof_authority") is False and probe.get("completion_authority") is False)
        _digest(probe["response_sha256"])
    return receipt


def _retained(parent, receipt):
    from ipfs_accelerate_py.agent_supervisor.runtime import owner_candidate_dispatch_audit as archive
    from ipfs_accelerate_py.agent_supervisor.runtime.source_formalization_context import _protected_directory
    from .terminal_context_audit import _open_directory, _read_at, _json, _same_json

    _need(parent is not None and os.getuid() == os.geteuid() and os.getuid() > 0)
    parent = Path(parent)
    _protected_directory(parent, owner=os.geteuid(), group=os.geteuid())
    dispatch = receipt["candidate_round_dispatch"]
    directory = archive.audit_directory_name(dispatch["session_id"])
    manifest_name = archive.audit_manifest_name(dispatch["session_id"])
    fd = _open_directory(parent)
    try:
        info = os.stat(directory, dir_fd=fd, follow_symlinks=False)
        _need(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700
            and info.st_uid == info.st_gid == os.geteuid())
        def read(name, limit):
            _need(type(name) is str and (name == manifest_name or
                name.startswith(directory + "/") and len(Path(name).parts) == 2))
            before = os.stat(name, dir_fd=fd, follow_symlinks=False)
            _need(stat.S_ISREG(before.st_mode) and stat.S_IMODE(before.st_mode) == 0o400
                and before.st_uid == before.st_gid == os.geteuid() and before.st_nlink == 1)
            raw = _read_at(fd, name, limit=limit)
            after = os.stat(name, dir_fd=fd, follow_symlinks=False)
            fields = ("st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_size", "st_mtime_ns", "st_ctime_ns")
            _need(all(getattr(before, key) == getattr(after, key) for key in fields))
            return raw
        raw = read(manifest_name, 262144)
        manifest = _json(raw)
        _need(type(manifest) is dict and manifest.get("schema") == archive.SCHEMA
            and all(type(manifest.get(k)) is type(v) and manifest[k] == v for k, v in archive.FLAGS.items()))
        binding = manifest["binding"]
        for key in ("round_number", "task_id", "native_prompt_sha256", "prior_session_id", "prior_descriptor_sha256",
            "session_id", "descriptor_sha256", "workspace_generation", "round_context_sha256",
            "candidate_scope_cid", "candidate_semantic_root_cid", "producer_binding_sha256", "handoff_sha256"):
            _need(_same_json(binding[key], dispatch[key]))
        _need(binding["candidate_input_sha256"] == receipt["prompt_sha256"])
        rows = manifest["artifacts"]
        _need(type(rows) is dict and set(rows) == set(archive.FILENAMES))
        bodies, total = {}, 0
        for role, row in rows.items():
            _need(type(row) is dict and set(row) == {"path", "sha256", "bytes"}
                and row["path"] == directory + "/" + archive.FILENAMES[role]
                and type(row["bytes"]) is int and 0 <= row["bytes"] <= 16 * 1024 * 1024)
            total += row["bytes"]
            _need(total <= 24 * 1024 * 1024)
            body = read(row["path"], row["bytes"])
            _need(len(body) == row["bytes"] and _sha(body) == row["sha256"])
            bodies[role] = body
        sources = manifest["candidate_sources"]
        context = _json(bodies["context"])
        _need(type(sources) is dict and set(sources) == set(context["source_hashes"]) and len(sources) <= 768)
        source_bytes = 0
        for index, (name, row) in enumerate(sorted(sources.items())):
            _need(type(row) is dict and set(row) == {"path", "sha256", "bytes"}
                and row["path"] == directory + "/source-" + str(index).zfill(4) + ".bin"
                and row["sha256"] == context["source_hashes"][name]
                and type(row["bytes"]) is int and 0 <= row["bytes"] <= 1024 * 1024)
            source_bytes += row["bytes"]
            _need(source_bytes <= 1024 * 1024 and total + source_bytes <= 24 * 1024 * 1024)
            body = read(row["path"], row["bytes"])
            _need(len(body) == row["bytes"] and _sha(body) == row["sha256"])
        final = os.stat(directory, dir_fd=fd, follow_symlinks=False)
        _need((info.st_dev, info.st_ino, info.st_mode, info.st_uid, info.st_gid) ==
              (final.st_dev, final.st_ino, final.st_mode, final.st_uid, final.st_gid))
        _protected_directory(parent, owner=os.geteuid(), group=os.geteuid())
        return bodies, manifest
    finally:
        os.close(fd)


def replay(*, rendered, receipt, parent):
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt, _candidate_round_guidance
    from ipfs_accelerate_py.agent_supervisor.runtime.candidate_reply_contract import apply_candidate_reply_contract
    from .terminal_context_audit import _json, _same_json

    validate_receipt(receipt)
    bodies, manifest = _retained(parent, receipt)
    dispatch = receipt["candidate_round_dispatch"]
    handoff, context, wire, descriptor, requirements = (_json(bodies[key]) for key in
        ("handoff", "context", "candidate_input", "descriptor", "original_requirements"))
    _need(_sha(bodies["handoff"]) == dispatch["handoff_sha256"]
        and _sha(bodies["descriptor"]) == dispatch["descriptor_sha256"]
        and _sha(bodies["context"]) == dispatch["round_context_sha256"]
        and _sha(bodies["candidate_input"]) == receipt["prompt_sha256"]
        and bodies["native_prompt"] == rendered.encode()
        and wire["native_prompt"] == rendered
        and wire["original_public_instruction_block"].encode() == bodies["original_public_instruction"]
        and wire["original_requirements_binding"]["original_requirements_sha256"] == _sha(bodies["original_requirements"])
        and requirements["native_prompt_sha256"] == receipt["native_prompt_sha256"]
        and requirements["public_instruction_block_sha256"] == _sha(bodies["original_public_instruction"])
        and descriptor["workspace"] == handoff["workspace"] == receipt["workspace"]
        and handoff["admission"] == _json(bodies["admission"])
        and handoff["prompt_observation"] == _json(bodies["prompt_observation"]))
    # The immutable handoff is the independently dispatched input binding.
    # Rehashing archived bytes and a report cannot change what it admitted.
    admitted = handoff["admission"]
    for key in ("round_number", "task_id", "native_prompt_sha256", "prior_session_id",
            "prior_descriptor_sha256", "session_id", "descriptor_sha256", "workspace_generation",
            "round_context_sha256", "candidate_scope_cid", "candidate_semantic_root_cid"):
        _need(_same_json(admitted[key], dispatch[key]))
        if key != "descriptor_sha256":
            _need(_same_json(descriptor[key], dispatch[key]))
    _need(handoff["provider_input_sha256"] == admitted["candidate_prompt_sha256"] == _sha(bodies["candidate_input"])
        and handoff["provider_input_bytes"] == len(bodies["candidate_input"])
        and handoff["descriptor_sha256"] == _sha(bodies["descriptor"])
        and handoff["native_prompt_sha256"] == _sha(bodies["native_prompt"])
        and handoff["native_prompt_bytes"] == len(bodies["native_prompt"])
        and _same_json(handoff["baseline_binding"], wire["baseline_binding"])
        and _same_json(handoff["baseline_binding"], context["baseline_binding"])
        and _same_json(handoff["candidate_artifacts"], context["artifacts"])
        and _same_json(wire["candidate_artifact_references"], context["artifacts"])
        and _same_json(context["artifacts"], dispatch["candidate_artifacts"])
        and handoff["prompt_observation"]["provider_prompt_sha256"] == _sha(bodies["candidate_input"])
        and handoff["prompt_observation"]["original_requirements_sha256"] == _sha(bodies["original_requirements"])
        and handoff["prompt_observation"]["public_instruction_block_sha256"] == _sha(bodies["original_public_instruction"]))
    for key in ("source_hashes", "program_paths", "producer_binding_sha256"):
        _need(_same_json(handoff[key], context[key]) and _same_json(wire["candidate_binding"][key], context[key]))
    for key in ("workspace_generation", "candidate_scope_cid", "semantic_root_cid"):
        _need(_same_json(wire["candidate_binding"][key], context[key]))
    for key in ("summary", "capsules", "formal"):
        _need(context["artifacts"][key] == dict(sha256=_sha(bodies[key]), bytes=len(bodies[key])))
    router = bodies["candidate_input"].decode() + _candidate_round_guidance(dispatch)
    model, advice = render_model_prompt(prompt=router, purpose="coding", workspace=Path(receipt["workspace"]), candidate_transport=True)
    model, reply = apply_candidate_reply_contract(model_prompt=model, binding=dispatch)
    checks = dict(retained_owner_input_bindings=True, retained_native_prompt=True,
        router_prompt_sha256=_sha(router.encode()) == receipt["router_prompt_sha256"],
        router_prompt_bytes=len(router.encode()) == receipt["router_prompt_bytes"],
        model_prompt_sha256=_sha(model.encode()) == receipt["model_prompt_sha256"],
        model_prompt_bytes=len(model.encode()) == receipt["model_prompt_bytes"],
        workspace_advisory_sha256=_sha(advice.encode()) == receipt["workspace_advisory_sha256"],
        workspace_advisory_bytes=len(advice.encode()) == receipt["workspace_advisory_bytes"],
        candidate_reply_contract=all(_same_json(value, receipt["candidate_reply_contract"][key])
            for key, value in reply.items() if key != "response_validated"))
    return model, advice, checks
