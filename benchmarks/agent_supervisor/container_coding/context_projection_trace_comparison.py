"""Replay an existing metadata projection over complete scripted model inputs.

This compares retained UTF-8 prompt and default source-client output bytes. It
reuses the production lossless codec and complete-input selector; a submitted
projection digest cannot nominate replacement content. Neither owner replay
records nor input controls are authenticated here. Native provider usage and
upstream source admission are outside this offline comparison's claims.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import source_view_trace_comparison as source

SCHEMA = "context-projection-trace-comparison@1"
CONTROLS_SCHEMA = "context-projection-trace-controls@1"
EXTRA_CONTROLS = frozenset({"baseline_metadata_mode", "treatment_metadata_mode",
    "source_view_profile", "provider", "coding_reply_mode", "workspace"})
TraceComparisonError = source.TraceComparisonError
canonical_json = source.canonical_json


def _controls(value):
    source._need(type(value) is dict and value.get("schema") == CONTROLS_SCHEMA
                 and EXTRA_CONTROLS <= set(value), "explicit context projection controls required")
    legacy = {key: item for key, item in value.items() if key not in EXTRA_CONTROLS}
    legacy["schema"] = source.CONTROLS_SCHEMA
    source._controls(legacy)
    source._need(value["baseline_metadata_mode"] == "legacy"
                 and value["treatment_metadata_mode"] == "common-bindings@1"
                 and value["source_view_profile"] == "identity@1",
                 "legacy/common-bindings pair with the same identity source reader required")
    source._need(value["provider"] in {"codex_cli", "grok_cli"}, "explicit coding provider required")
    from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import validate_coding_reply_mode
    validate_coding_reply_mode(mode=value["coding_reply_mode"], purpose="coding", provider=value["provider"])
    workspace = value["workspace"]
    source._text(workspace, empty=False)
    source._need(Path(workspace).is_absolute() and str(Path(workspace)) == workspace
                 and ".." not in Path(workspace).parts and "\\" not in workspace
                 and all(ord(c) >= 32 and ord(c) != 127 for c in workspace),
                 "canonical captured absolute workspace required")
    return value


def _projection(baseline_prompt, treatment_prompt, controls):
    from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata
    from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
    from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_client import guidance

    rendering = dict(purpose="coding", workspace=Path(controls["workspace"]), semantic_transport=True)
    _, advisory = render_model_prompt(prompt="", **rendering)
    source._need(baseline_prompt.startswith(advisory), "baseline lacks the production workspace advisory")
    body = baseline_prompt[len(advisory):]
    # The production reply mode appends one fixed instruction after all context.
    reply, _ = apply_coding_reply_contract(model_prompt="", mode=controls["coding_reply_mode"],
                                           purpose="coding", provider=controls["provider"])
    if reply:
        source._need(body.endswith(reply) and body.count(reply) == 1,
                     "baseline lacks the exact single production reply contract")
        body = body[:-len(reply)]
    wire, end = json.JSONDecoder().raw_decode(body)
    native_transport, suffix = body[:end], body[end:]
    source._need(type(wire) is dict and wire.get("schema") == metadata.TRANSPORT_SCHEMA,
                 "canonical original semantic transport required")
    # The real codec checks canonical JSON, closed transport fields, and exact
    # equality after restoring its projected candidate, including raw sources.
    view = metadata.project_semantic_metadata_view(native_transport)
    semantic = wire["translated_semantic"]
    manifest = semantic["manifest"]
    source._need(all(type(manifest.get(path)) is dict and manifest[path].get("sha256") == digest
                     for path, digest in controls["source_hashes"].items()),
                 "captured source inputs differ from semantic transport manifest")
    advice = guidance(controls["source_view_profile"])
    source._need(suffix.count(advice) == 1 and baseline_prompt.count(advice) == 1,
                 "exact source reader guidance required once outside transport")
    reconstructed, _ = render_model_prompt(prompt=native_transport + suffix, **rendering)
    reconstructed, _ = apply_coding_reply_contract(model_prompt=reconstructed,
        mode=controls["coding_reply_mode"], purpose="coding", provider=controls["provider"])
    source._need(reconstructed == baseline_prompt, "baseline complete rendering differs")
    candidate, _ = render_model_prompt(prompt=view.provider_prompt + suffix, **rendering)
    candidate, _ = apply_coding_reply_contract(model_prompt=candidate,
        mode=controls["coding_reply_mode"], purpose="coding", provider=controls["provider"])
    selection = metadata.select_semantic_metadata_view(view=view,
        native_complete_prompt=baseline_prompt, candidate_complete_prompt=candidate)
    source._need(treatment_prompt == selection.selected_prompt,
                 "captured treatment differs from independently replayed complete-input selection")
    if selection.selected_mode != "legacy":
        source._need(len(treatment_prompt.encode()) < len(baseline_prompt.encode()),
                     "selected metadata projection must reduce actual complete prompt bytes")
    return dict(
        requested_mode=controls["treatment_metadata_mode"], selected_mode=selection.selected_mode,
        fallback_reason=selection.receipt["fallback_reason"],
        native_transport_sha256=source._sha(native_transport.encode()),
        native_transport_bytes=len(native_transport.encode()),
        candidate_transport_sha256=source._sha(view.provider_prompt.encode()),
        candidate_transport_bytes=len(view.provider_prompt.encode()),
        view_receipt_sha256=source._sha(view.receipt_json.encode()),
        selection_receipt_sha256=source._sha(selection.receipt_json.encode()),
        workspace_advisory_sha256=source._sha(advisory.encode()), workspace_advisory_bytes=len(advisory.encode()),
        unchanged_router_suffix_sha256=source._sha(suffix.encode()), unchanged_router_suffix_bytes=len(suffix.encode()),
        coding_reply_instruction_sha256=source._sha(reply.encode()), coding_reply_instruction_bytes=len(reply.encode()),
        source_guidance_sha256=source._sha(advice.encode()), source_guidance_bytes=len(advice.encode()),
        source_manifest_sha256=view.receipt["source_manifest_sha256"],
        capsule_count=view.receipt["capsule_count"], admission_count=view.receipt["admission_count"],
        raw_source_values_preserved=True, exact_transport_round_trip_verified=True,
        complete_prompt_reconstruction_verified=True, selection_basis="utf8-bytes-and-existing-proxy@1",
        provider_tokens_measured=False, upstream_source_admission_authenticated=False), manifest


def _catalog_facts(trace, operation_id, manifest):
    """Retain catalog facts while binding the enumerated generation-scoped IDs.

    Each original artifact has already passed exact owner replay in _arm. This
    normalizes only identities scoped to the independent reader session. It is
    not a general semantic equivalence checker or an authenticated owner replay.
    """
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_artifacts import AUTHORITY
    from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
    from ipfs_datasets_py.logic.software_contracts.content import validate_cid
    pages = [source._loads(row["stdout"]) for row in trace["calls"] if row["operation_id"] == operation_id]
    value = source._loads(b"".join(source._payload(page) for page in pages).decode())
    source._need(type(value) is dict and value.get("schema") == "supervisor-worker-candidate-selection@1"
                 and value.get("artifact") == "catalog" and value.get("source_state") == "working-candidate",
                 "generation-scoped native catalog required")
    source._need(all(type(value.get(k)) is type(v) and value[k] == v for k, v in AUTHORITY.items()),
                 "catalog nonproof labels differ")
    binding = value.get("binding")
    source._fields(binding, {"candidate_scope_cid", "workspace_generation", "semantic_root_cid",
        "parent_artifact_sha256", "source_hashes_sha256", "producer_binding_sha256"})
    generation = binding["workspace_generation"]
    source._need(generation in trace["generations"].values()
                 and all(page["workspace_generation"] == generation for page in pages),
                 "catalog generation differs from captured source session")
    role = next(name for name, digest in trace["generations"].items() if digest == generation)
    source._need(source._digest(binding["source_hashes_sha256"])
                 and source._digest(binding["producer_binding_sha256"]), "catalog digest binding differs")
    parents = binding["parent_artifact_sha256"]
    source._fields(parents, {"summary", "formal", "capsules"})
    source._need(all(source._digest(digest) for digest in parents.values()), "catalog parent digests differ")
    rows = value.get("rows")
    source._need(type(rows) is list and 1 <= len(rows) <= 64
                 and all(type(row) is dict for row in rows)
                 and [row.get("path") for row in rows] == sorted(manifest), "complete catalog source inventory required")
    hashes, programs = {}, []
    seen_symbols = set()
    for row in rows:
        path = row["path"]
        source._need(row.get("source_sha256") == manifest[path]["sha256"]
                     and source._integer(row.get("source_bytes")) and row.get("role") in {"program", "support"},
                     "catalog captured source binding differs")
        hashes[path] = row["source_sha256"]
        if row["role"] == "program":
            programs.append(path)
        capsules = row.get("capsules")
        source._need(type(capsules) is list and type(row.get("capsule_count")) is int
                     and row["capsule_count"] == len(capsules), "catalog capsule counts differ")
        normalized = []
        for capsule in capsules:
            source._fields(capsule, {"stable_symbol_id", "capsule_cid", "names", "kinds"})
            for field in ("stable_symbol_id", "capsule_cid"):
                validate_cid(capsule[field])
            source._need(capsule["stable_symbol_id"] not in seen_symbols, "duplicate catalog symbol identity")
            seen_symbols.add(capsule["stable_symbol_id"])
            for field in ("names", "kinds"):
                source._need(type(capsule[field]) is list and all(type(item) is str for item in capsule[field])
                             and capsule[field] == sorted(set(capsule[field])), "catalog capsule facts differ")
            normalized.append({key: capsule[key] for key in ("names", "kinds")})
        row["capsules"] = sorted(normalized, key=canonical_json)
    source._need(binding["source_hashes_sha256"] == source._sha(canonical_json(hashes)),
                 "catalog source inventory digest differs")
    expected_scope = cid_for_payload(dict(schema="supervisor-worker-candidate-scope@1",
        workspace_generation=generation, source_hashes=hashes, program_paths=programs))
    source._need(binding["candidate_scope_cid"] == expected_scope, "catalog candidate scope binding differs")
    validate_cid(binding["semantic_root_cid"])
    binding.update(workspace_generation=role, candidate_scope_cid=None, semantic_root_cid=None,
                   parent_artifact_sha256={key: None for key in parents})
    return canonical_json(value)


def _same_operations(baseline, treatment, left, right, controls, manifest):
    source._need(len(left["complete_operations"]) == len(right["complete_operations"]),
                 "complete operation coverage differs")
    kinds = {row["operation_id"]: row["kind"] for row in controls["logical_operations"]}
    normalized = []
    for before, after in zip(left["complete_operations"], right["complete_operations"]):
        source._need(before["operation_id"] == after["operation_id"], "complete operation ordering differs")
        if before == after:
            continue
        operation = before["operation_id"]
        source._need(kinds[operation] == "catalog", "matched exact source output differs")
        a = _catalog_facts(baseline, operation, manifest)
        b = _catalog_facts(treatment, operation, manifest)
        source._need(a == b, "catalog facts differ beyond generation-scoped identities")
        normalized.append(dict(operation_id=operation, facts_sha256=source._sha(a),
            normalization="enumerated-generation-scoped-catalog-identities@1",
            original_artifacts_independently_owner_replayed=True, owner_replay_authenticated=False))
    return normalized


def compare_traces(*, baseline, treatment, controls, baseline_replay, treatment_replay):
    """Compare the same complete source workload with lossless metadata factoring.

    The source reader profile, public source bytes and edit are fixed. The only
    permitted model-input change is the production metadata codec's selected
    candidate, with the original full wrapper, guidance and constraints retained.
    A nonshrinking candidate must be observed as the native-input fallback.
    """
    try:
        controls = _controls(controls)
        source._need(baseline["profile"] == treatment["profile"] == controls["source_view_profile"],
                     "source reader profile differs across metadata arms")
        left = source._arm(baseline, baseline_replay, controls)
        right = source._arm(treatment, treatment_replay, controls)
        for trace in (baseline, treatment):
            source._need(trace.get("workspace") == controls["workspace"], "captured workspace differs from rendering controls")
        for kind in {row["kind"] for row in baseline["calls"]}:
            clients = {tuple(row["argv"][:4]) for trace in (baseline, treatment)
                       for row in trace["calls"] if row["kind"] == kind}
            source._need(len(clients) == 1, "client interpreter/path controls differ")
        for field in ("refusals", "operation_controls"):
            source._need(left[field] == right[field], "matched source workload or generation policies differ")
        for field in ("uid", "gid", "task_alias"):
            source._need(baseline.get(field) == treatment.get(field), "worker execution controls differ")
        projection, manifest = _projection(baseline["prompt"]["text"], treatment["prompt"]["text"], controls)
        catalog_comparisons = _same_operations(baseline, treatment, left, right, controls, manifest)
        return dict(schema=SCHEMA, controls_sha256=source._sha(canonical_json(controls)),
            baseline=left, treatment=right, projection=projection, catalog_comparisons=catalog_comparisons,
            baseline_minus_treatment_scripted_exposure_bytes=left["scripted_exposure_bytes"] - right["scripted_exposure_bytes"],
            scope="one-pass prepared prompt plus captured default CLI stdout/stderr; scripted argv and transport JSON separate",
            provider_tokens=None, official_reward=None, provider_received_input=False, provider_calls=0, model_calls=0,
            total_token_savings_measured=False, owner_receipts_authenticated=False,
            source_obligations_discharged=0, proof_authority=False, completion_authority=False,
            mapped_edits_compared=False)
    except TraceComparisonError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise TraceComparisonError("invalid captured metadata trace") from exc


__all__ = ["compare_traces", "canonical_json", "TraceComparisonError", "SCHEMA", "CONTROLS_SCHEMA"]
