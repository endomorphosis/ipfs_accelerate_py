"""Exact full-input replay and source-workload pairing; no provider calls."""
from copy import deepcopy
import json

import pytest

from benchmarks.agent_supervisor.container_coding import context_projection_trace_comparison as comparison
from benchmarks.agent_supervisor.container_coding import source_view_trace_comparison as source
from benchmarks.agent_supervisor.container_coding.test_source_view_trace_comparison import (
    fixture as source_fixture, sha, streams, wire,
)
from ipfs_accelerate_py.agent_supervisor.runtime import semantic_metadata_view as metadata
from ipfs_accelerate_py.agent_supervisor.runtime.coding_reply_contract import apply_coding_reply_contract
from ipfs_accelerate_py.agent_supervisor.runtime.router_implementation_runner import render_model_prompt
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_client import guidance
from pathlib import Path


def reseal(value):
    for arm in ("baseline", "treatment"):
        trace, replay = value[arm], value[arm + "_replay"]
        prompt = trace["prompt"]["text"].encode()
        trace["prompt"].update(sha256=sha(prompt), utf8_bytes=len(prompt))
        replay.update(trace_sha256=sha(wire(trace)), prompt_sha256=sha(prompt),
                      controls_sha256=sha(wire(value["controls"])))


def fixture(*, count=6, reply_mode="ordinary-completion@1", provider="codex_cli", paginate=False):
    value = source_fixture(shrink=False, paginate=paginate)
    controls = value["controls"]
    controls.update(schema=comparison.CONTROLS_SCHEMA, baseline_metadata_mode="legacy",
        treatment_metadata_mode="common-bindings@1", source_view_profile="identity@1",
        provider=provider, coding_reply_mode=reply_mode, workspace="/allocated/workspace")
    original = 'idx = "λ"  \r\nanswer = idx  \r\n'
    # Authored shape for the pure codec. Native producer freshness is deliberately
    # not claimed by this helper; existing codec integration and Docker test cover it.
    native = dict(schema=metadata.TRANSPORT_SCHEMA, translation_cid="translation:authored",
        translation_table={}, native_context=dict(objective_id="authored-task", evidence=[]),
        translated_semantic=dict(task_id="authored-task", scope_cid="scope:authored", semantic_root_cid="root:authored",
            manifest={"m.py": dict(sha256=controls["source_hashes"]["m.py"], source_cid="source:authored")},
            raw_sources={"m.py": original},
            capsules=[dict(schema="authored-capsule@1", capsule_schema="authored-capsule@1",
                semantic_index_schema="authored-index@1", capsule_compiler_version="authored@1",
                extractor_version="authored@1", source_cid="source:" + "a" * 60,
                stable_symbol_id="authored:" + str(i), signature="f(x)", metadata=dict(pending=True)) for i in range(count)],
            admissions=[dict(schema="authored-admission@1", interface="authored-interface@1", freshness="current",
                ref=dict(source_cid="source:" + "a" * 60, semantic_state_root_cid="root:" + "b" * 60,
                    raw_source_required=True, stable_symbol_id="authored:" + str(i))) for i in range(count)]),
        native_suffix="Literal task suffix λ\r\n", instructions="Authored retained instructions.",
        execution_authority=False, completion_authority=False)
    native_text = wire(native).decode()
    projected = metadata.project_semantic_metadata_view(native_text)
    suffix = guidance("identity@1") + "\nLiteral unchanged authorization and uncertainty λ\r\n"
    prompts = []
    for text in (native_text, projected.provider_prompt):
        prompt, _ = render_model_prompt(prompt=text + suffix, purpose="coding",
            workspace=Path(controls["workspace"]), semantic_transport=True)
        prompt, _ = apply_coding_reply_contract(model_prompt=prompt, mode=reply_mode, purpose="coding", provider=provider)
        prompts.append(prompt)
    selected = metadata.select_semantic_metadata_view(view=projected,
        native_complete_prompt=prompts[0], candidate_complete_prompt=prompts[1])
    for arm, prompt in (("baseline", prompts[0]), ("treatment", selected.selected_prompt)):
        value[arm].update(profile="identity@1", workspace=controls["workspace"])
        value[arm]["prompt"]["text"] = prompt
    reseal(value)
    return value


def test_complete_actual_prompt_replayed_with_all_wrappers_and_source_reads():
    value = fixture()
    result = comparison.compare_traces(**value)
    left, right, projection = result["baseline"], result["treatment"], result["projection"]
    assert projection["selected_mode"] == "common-bindings@1"
    assert left["prepared_prompt_bytes"] > right["prepared_prompt_bytes"]
    assert left["stdout_bytes"] == right["stdout_bytes"]
    assert result["baseline_minus_treatment_scripted_exposure_bytes"] == left["prepared_prompt_bytes"] - right["prepared_prompt_bytes"]
    assert left["complete_operations"] == right["complete_operations"]
    assert projection["raw_source_values_preserved"] and projection["exact_transport_round_trip_verified"]
    assert projection["source_guidance_bytes"] == len(guidance("identity@1").encode())
    assert result["provider_tokens"] is result["official_reward"] is None
    assert not result["total_token_savings_measured"] and not result["owner_receipts_authenticated"]
    assert not projection["upstream_source_admission_authenticated"]
    assert b"Literal unchanged authorization" not in wire(result)
    assert b'idx = ' not in wire(result)


def test_nonshrinking_projection_requires_actual_legacy_fallback():
    value = fixture(count=1)
    result = comparison.compare_traces(**value)
    assert result["projection"]["selected_mode"] == "legacy"
    assert result["projection"]["fallback_reason"] == "complete_input_not_smaller_under_bytes_and_proxy"
    assert result["baseline_minus_treatment_scripted_exposure_bytes"] == 0


@pytest.mark.parametrize("provider", ["codex_cli", "grok_cli"])
def test_legacy_reply_mode_keeps_existing_provider_route(provider):
    result = comparison.compare_traces(**fixture(reply_mode="legacy", provider=provider))
    assert result["projection"]["coding_reply_instruction_bytes"] == 0
    assert result["projection"]["selected_mode"] == "common-bindings@1"


def test_different_complete_pagination_and_actual_stderr_are_charged():
    value = fixture(paginate=True)
    streams(value["treatment"]["calls"][0], stderr="authored diagnostic λ\r\n")
    reseal(value)
    result = comparison.compare_traces(**value)
    assert result["baseline"]["calls"] == result["treatment"]["calls"] + 1
    assert result["treatment"]["stderr_bytes"] == len("authored diagnostic λ\r\n".encode())
    assert result["baseline"]["logical_operations"] == result["treatment"]["logical_operations"]


@pytest.mark.parametrize("change", ["treatment_suffix", "treatment_raw_source", "treatment_native_core", "treatment_schema",
    "baseline_advisory", "baseline_reply", "duplicate_reply", "missing_guidance", "duplicate_guidance",
    "noncanonical_transport", "duplicate_json", "transport_v2", "unknown_transport_field", "semantic_source_hash",
    "profile", "controls_profile", "controls_mode", "controls_extra", "controls_provider", "controls_workspace",
    "trace_workspace", "call_binary", "operation_policy", "source_postimage", "owner_cleanup", "owner_controls",
    "source_page", "proof_claim", "stream_newline", "dropped_call", "unchanged_treatment", "extra_treatment_discount"])
def test_resealed_worker_or_controls_cannot_bypass_actual_projection_and_workload(change):
    value = fixture()
    trace = value["treatment"]
    prompt = trace["prompt"]
    if change == "treatment_suffix": prompt["text"] += "unauthorized extra context"
    elif change == "treatment_raw_source": prompt["text"] = prompt["text"].replace('idx =', 'lost =', 1)
    elif change == "treatment_native_core": prompt["text"] = prompt["text"].replace('"objective_id":"authored-task"', '"objective_id":"foreign"')
    elif change == "treatment_schema": prompt["text"] = prompt["text"].replace(metadata.VIEW_SCHEMA, metadata.TRANSPORT_SCHEMA)
    elif change == "baseline_advisory": value["baseline"]["prompt"]["text"] = value["baseline"]["prompt"]["text"].replace("Do not write to canonical /app.", "Write anywhere.")
    elif change == "baseline_reply": value["baseline"]["prompt"]["text"] = value["baseline"]["prompt"]["text"].replace("Return only", "Return prose") + "extra"
    elif change == "duplicate_reply":
        reply, _ = apply_coding_reply_contract(model_prompt="", mode="ordinary-completion@1", purpose="coding", provider="codex_cli")
        value["baseline"]["prompt"]["text"] += reply
    elif change in {"missing_guidance", "duplicate_guidance"}:
        for arm in ("baseline", "treatment"):
            value[arm]["prompt"]["text"] = value[arm]["prompt"]["text"].replace(guidance("identity@1"),
                "" if change == "missing_guidance" else guidance("identity@1") * 2)
    elif change in {"noncanonical_transport", "duplicate_json", "transport_v2", "unknown_transport_field", "semantic_source_hash"}:
        base = value["baseline"]["prompt"]
        if change == "noncanonical_transport": base["text"] = base["text"].replace('{"completion_authority":false', '{ "completion_authority":false', 1)
        elif change == "duplicate_json": base["text"] = base["text"].replace('{"completion_authority":false', '{"completion_authority":false,"completion_authority":false', 1)
        elif change == "transport_v2": base["text"] = base["text"].replace(metadata.TRANSPORT_SCHEMA, "supervisor-semantic-router-input@2")
        elif change == "unknown_transport_field": base["text"] = base["text"].replace('{"completion_authority":false', '{"invented":true,"completion_authority":false', 1)
        else: base["text"] = base["text"].replace(value["controls"]["source_hashes"]["m.py"], "0" * 64)
    elif change == "profile": trace["profile"] = "python-comments@1"
    elif change == "controls_profile": value["controls"]["source_view_profile"] = "python-comments@1"
    elif change == "controls_mode": value["controls"]["treatment_metadata_mode"] = "common-bindings-instruction@1"
    elif change == "controls_extra": value["controls"]["projection_sha256"] = prompt["sha256"]
    elif change == "controls_provider": value["controls"]["provider"] = "grok_cli"
    elif change == "controls_workspace": value["controls"]["workspace"] = "/allocated/other"
    elif change == "trace_workspace": trace["workspace"] = "/allocated/other"
    elif change == "call_binary": trace["calls"][0]["argv"][0] = "/other/python"
    elif change == "operation_policy":
        argv = trace["calls"][0]["argv"]
        index = argv.index("--expected-workspace-generation")
        del argv[index:index + 2]
    elif change == "source_postimage": trace["edited_source_hashes"]["m.py"] = "0" * 64
    elif change == "owner_cleanup": value["treatment_replay"]["cleanup_verified"] = False
    elif change == "owner_controls": value["treatment_replay"]["schema"] = "not-an-owner-replay"
    elif change == "source_page":
        page = json.loads(trace["calls"][0]["stdout"])
        page["content"] = "different"
        streams(trace["calls"][0], stdout=wire(page).decode() + "\n")
    elif change == "proof_claim": trace["source_obligations_discharged"] = 1
    elif change == "stream_newline": streams(trace["calls"][0], stdout=trace["calls"][0]["stdout"].rstrip("\n"))
    elif change == "dropped_call": trace["calls"].pop()
    elif change == "unchanged_treatment": prompt["text"] = value["baseline"]["prompt"]["text"]
    else: trace["discount_bytes"] = 1000
    reseal(value)
    with pytest.raises(comparison.TraceComparisonError):
        comparison.compare_traces(**value)


def test_nonshrinking_candidate_cannot_be_forced_by_claiming_treatment_mode():
    value = fixture(count=1)
    base = value["baseline"]["prompt"]["text"]
    _, advisory = render_model_prompt(prompt="", purpose="coding", workspace=Path(value["controls"]["workspace"]), semantic_transport=True)
    body = base[len(advisory):]
    _record, end = json.JSONDecoder().raw_decode(body)
    projected = metadata.project_semantic_metadata_view(body[:end])
    value["treatment"]["prompt"]["text"] = advisory + projected.provider_prompt + body[end:]
    reseal(value)
    with pytest.raises(comparison.TraceComparisonError, match="replayed complete-input selection"):
        comparison.compare_traces(**value)


def catalog_fixture():
    """Separate session identities change while native catalog facts stay fixed."""
    from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_artifacts import AUTHORITY
    from ipfs_accelerate_py.agent_supervisor.semantic_state.wire import cid_for_payload
    value = fixture()
    catalogs = {}
    for arm in ("baseline", "treatment"):
        trace = value[arm]
        generation = trace["generations"]["baseline"]
        hashes = trace["source_hashes"]
        capsules = [dict(stable_symbol_id=cid_for_payload(dict(arm=arm, symbol=index)),
            capsule_cid=cid_for_payload(dict(arm=arm, capsule=index)), names=["name" + str(index)], kinds=["FunctionDef"])
            for index in range(2)]
        if arm == "treatment": capsules.reverse()
        catalogs[arm] = dict(schema="supervisor-worker-candidate-selection@1", artifact="catalog",
            source_state="working-candidate", binding=dict(workspace_generation=generation,
                candidate_scope_cid=cid_for_payload(dict(schema="supervisor-worker-candidate-scope@1",
                    workspace_generation=generation, source_hashes=hashes, program_paths=["m.py"])),
                semantic_root_cid=cid_for_payload(dict(arm=arm)),
                source_hashes_sha256=sha(wire(hashes)), producer_binding_sha256=sha(b"producer"),
                parent_artifact_sha256={key: sha((arm + key).encode()) for key in ("formal", "summary", "capsules")}),
            rows=[dict(path="m.py", role="program", source_sha256=hashes["m.py"], source_bytes=32,
                capsule_count=2, capsules=capsules, pending_obligations=7)], **AUTHORITY)
    replace_catalogs(value, catalogs)
    return value, catalogs


def replace_catalogs(value, catalogs):
    for arm, catalog in catalogs.items():
        raw = wire(catalog)
        call = next(row for row in value[arm]["calls"] if row["kind"] == "catalog")
        page = json.loads(call["stdout"])
        page.update(content=raw.decode(), artifact_bytes=len(raw), end_offset=len(raw), artifact_sha256=sha(raw))
        streams(call, stdout=wire(page).decode() + "\n")
        replay = value[arm + "_replay"]
        operation = next(row for row in replay["operation_replays"] if row["operation_id"] == call["operation_id"])
        operation.update(sha256=sha(raw), bytes=len(raw))
        replay["response_json_bytes"] = replay["delivered_response_json_bytes"] = sum(row["stdout_utf8_bytes"] + 1000 for row in value[arm]["calls"])
    reseal(value)


def test_independent_native_catalog_identities_are_normalized_without_discarding_facts():
    value, _ = catalog_fixture()
    result = comparison.compare_traces(**value)
    assert result["baseline"]["complete_operations"] != result["treatment"]["complete_operations"]
    assert len(result["catalog_comparisons"]) == 1
    normalized = result["catalog_comparisons"][0]
    assert normalized["normalization"] == "enumerated-generation-scoped-catalog-identities@1"
    assert normalized["original_artifacts_independently_owner_replayed"]
    assert not normalized["owner_replay_authenticated"]


@pytest.mark.parametrize("change", ["fact", "generation", "scope", "source_hash", "producer_hash", "capsule_fact",
    "capsule_duplicate", "capsule_identity", "parent_digest", "inventory", "authority", "count", "new_field"])
def test_catalog_normalization_never_hides_changed_facts_or_unbound_identity(change):
    value, catalogs = catalog_fixture()
    catalog = catalogs["treatment"]
    row = catalog["rows"][0]
    if change == "fact": row["pending_obligations"] = 0
    elif change == "generation": catalog["binding"]["workspace_generation"] = "0" * 64
    elif change == "scope": catalog["binding"]["candidate_scope_cid"] = catalog["binding"]["semantic_root_cid"]
    elif change == "source_hash": row["source_sha256"] = "0" * 64
    elif change == "producer_hash": catalog["binding"]["producer_binding_sha256"] = "0" * 64
    elif change == "capsule_fact": row["capsules"][0]["names"] = ["different"]
    elif change == "capsule_duplicate": row["capsules"][0]["stable_symbol_id"] = row["capsules"][1]["stable_symbol_id"]
    elif change == "capsule_identity": row["capsules"][0]["stable_symbol_id"] = "invalid"
    elif change == "parent_digest": catalog["binding"]["parent_artifact_sha256"]["formal"] = "invalid"
    elif change == "inventory": row["path"] = "foreign.py"
    elif change == "authority": catalog["proof_authority"] = True
    elif change == "count": row["capsule_count"] = True
    else: row["extra_fact"] = "new"
    replace_catalogs(value, catalogs)
    with pytest.raises(comparison.TraceComparisonError):
        comparison.compare_traces(**value)
