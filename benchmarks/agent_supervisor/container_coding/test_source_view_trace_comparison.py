"""Captured-output fidelity and matched-workload controls, without a provider."""
from copy import deepcopy
import base64
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import source_view_trace_comparison as comparison
from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_client import guidance


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def wire(value):
    return comparison.canonical_json(value)


def streams(call, stdout=None, stderr=None):
    if stdout is not None:
        call["stdout"] = stdout
    if stderr is not None:
        call["stderr"] = stderr
    for key in ("stdout", "stderr"):
        call[key + "_utf8_bytes"] = len(call[key].encode())
        call[key + "_sha256"] = sha(call[key].encode())


def fixture(*, shrink=True, paginate=False, binary_slices=False):
    original = 'idx = "λ"  \r\nanswer = idx  \r\n'.encode()
    edited = original.replace(b"idx", b"i")
    phases = [("bootstrap", "compact", "baseline", "complete"),
              ("discovery", None, "baseline", "complete"),
              ("pre_edit_expansion", "original", "baseline", "complete"),
              ("stale_generation", "compact", "baseline", "refused"),
              ("edited_refresh", "compact", "edited", "complete"),
              ("post_edit_expansion", "original", "edited", "complete"),
              ("restore_stale_generation", "compact", "edited", "refused"),
              ("restored_refresh", "compact", "restored", "complete")]
    operations = [dict(operation_id=phase + ":" + ("catalog" if view is None else "m.py"),
        kind="catalog" if view is None else "source", path=None if view is None else "m.py", view=view, outcome=outcome)
        for phase, view, _, outcome in phases]
    controls = dict(schema=comparison.CONTROLS_SCHEMA, common_input_sha256={"native_prompt": sha(b"native")},
        runtime_sha256={"fixture": sha(b"runtime")}, script_sha256=sha(b"script"), page_budget=8192,
        source_hashes={"m.py": sha(original)}, logical_operations=operations,
        source_edit=dict(path="m.py", old="idx", new="i", occurrences=2, before_sha256=sha(original), after_sha256=sha(edited)))
    result = dict(controls=controls)
    for label, profile in (("baseline", "identity@1"), ("treatment", "python-comments@1")):
        generations = {name: sha((label + name).encode()) for name in ("baseline", "edited", "restored")}
        source_hashes = {"m.py": sha(original)}
        prompt = "same required source: " + original.decode() + guidance(profile) + "same response contract"
        trace = dict(schema=comparison.TRACE_SCHEMA, profile=profile, calls=[],
            prompt=dict(text=prompt, utf8_bytes=len(prompt.encode()), sha256=sha(prompt.encode())),
            source_hashes=source_hashes, edited_source_hashes={"m.py": sha(edited)}, restored_source_hashes=source_hashes,
            generations=generations, provider_calls=0, model_calls=0, provider_tokens=None,
            source_obligations_discharged=0, total_token_savings_measured=False, edits_mapped=False,
            actual_original_source_edit=True, original_source_restored=True, full_program_behavior_verified=False)
        replays = []
        for operation, (phase, view, state, outcome) in zip(operations, phases):
            source = edited if state == "edited" else original
            compact = source.replace(b"  \r\n", b"\r\n") if shrink and label == "treatment" else source
            data = wire({"rows": [{"path": "m.py"}]}) if view is None else (source if view == "original" else compact)
            parts = [data]
            if paginate and label == "baseline" and phase == "bootstrap":
                parts = [data[:10], data[10:]]
            offset, previous = 0, None
            for page_index, part in enumerate(parts):
                argv = ["/opt/python", "-I", "-B", "/opt/worker_source_client.py" if view else "/opt/worker_candidate_context_client.py",
                        "--max-bytes", "8192"]
                if view is None:
                    argv += ["--select", "--artifact", "catalog"]
                else:
                    argv += ["--path", "m.py", "--view", view, "--working"]
                argv += ["--expected-workspace-generation", generations[state]]
                if view == "original" or previous is not None and view is not None:
                    argv += ["--expected-source-sha256", sha(source), "--expected-view-sha256", sha(compact)]
                if previous is not None:
                    argv += ["--cursor", json.dumps(previous)]
                if outcome == "refused":
                    page = dict(ok=False, reason_code="stale_context")
                else:
                    exhausted = page_index == len(parts) - 1
                    cursor = None if exhausted else dict(schema="supervisor-worker-source-view-cursor@1",
                        request_sha256=sha(b"query"), offset=offset + len(part), token="a" * 32)
                    page = dict(ok=True, offset=offset, end_offset=offset + len(part), exhausted=exhausted,
                        next_cursor=cursor, encoding="base64" if binary_slices else "utf-8",
                        content=base64.b64encode(part).decode() if binary_slices else part.decode(),
                        source_state="working-candidate", workspace_generation=generations[state], baseline_evidence_applicable=False)
                    if view is not None:
                        page.update(path="m.py", view=view, source_sha256=sha(source), view_sha256=sha(compact),
                                    baseline_source_sha256=sha(original))
                    else:
                        page.update(artifact="catalog", artifact_sha256=sha(data), artifact_bytes=len(data),
                            proof_authority=False, completion_authority=False, source_obligations_discharged=0,
                            selection=dict(path=None, symbol_id=None, function_id=None, kind=None))
                    previous, offset = cursor, offset + len(part)
                call = dict(index=len(trace["calls"]), operation_id=operation["operation_id"], page_index=page_index,
                    phase=phase, kind=operation["kind"], path=operation["path"], view=view, argv=argv,
                    returncode=0 if outcome == "complete" else 2, stdout="", stderr="")
                streams(call, wire(page).decode() + "\n")
                trace["calls"].append(call)
            if outcome == "complete":
                replays.append(dict(operation_id=operation["operation_id"], sha256=sha(data), bytes=len(data)))
        replay = dict(schema=comparison.REPLAY_SCHEMA, trace_sha256=sha(wire(trace)), controls_sha256=sha(wire(controls)),
            prompt_sha256=trace["prompt"]["sha256"], operation_replays=replays,
            **{key: trace[key] for key in ("source_hashes", "edited_source_hashes", "restored_source_hashes")},
            response_json_bytes=sum(c["stdout_utf8_bytes"] + 1000 for c in trace["calls"]),
            delivered_response_json_bytes=sum(c["stdout_utf8_bytes"] + 1000 for c in trace["calls"]),
            delivered_responses=len(trace["calls"]), accounting_complete=True, source_replay_verified=True,
            source_edit_verified=True, cleanup_verified=True)
        result[label], result[label + "_replay"] = trace, replay
    return result


def seal(value, arm="treatment"):
    value[arm + "_replay"]["trace_sha256"] = sha(wire(value[arm]))


def test_exact_utf8_stdout_newlines_stderr_and_argv_are_distinct_ledgers():
    value = fixture()
    call = value["treatment"]["calls"][0]
    streams(call, stderr="fixture diagnostic λ\r\n")
    seal(value)
    observed = comparison.compare_traces(**value)
    arm = observed["treatment"]
    expected_stdout = sum(len(c["stdout"].encode()) for c in value["treatment"]["calls"])
    assert arm["stdout_bytes"] == expected_stdout
    assert arm["stderr_bytes"] == len("fixture diagnostic λ\r\n".encode())
    assert arm["scripted_exposure_bytes"] == arm["prepared_prompt_bytes"] + expected_stdout + arm["stderr_bytes"]
    assert arm["scripted_argv_json_bytes"] == sum(len(wire(c["argv"])) for c in value["treatment"]["calls"])
    assert arm["transport_response_json_bytes"] > expected_stdout
    assert observed["provider_tokens"] is observed["official_reward"] is None
    assert observed["provider_received_input"] is observed["total_token_savings_measured"] is False
    assert b"same required source" not in wire(observed)


def test_different_page_counts_are_allowed_only_with_identical_logical_coverage():
    value = fixture(paginate=True, binary_slices=True)
    result = comparison.compare_traces(**value)
    assert result["baseline"]["calls"] == result["treatment"]["calls"] + 1
    assert result["baseline"]["logical_operations"] == result["treatment"]["logical_operations"]


def test_identity_fallback_retains_negative_result_from_additional_guidance():
    value = fixture(shrink=False)
    result = comparison.compare_traces(**value)
    assert result["baseline"]["stdout_bytes"] == result["treatment"]["stdout_bytes"]
    assert result["baseline_minus_treatment_scripted_exposure_bytes"] < 0


@pytest.mark.parametrize("change", ["stream_hash", "stream_size", "bool_size", "newline", "raw_response", "extra_field",
    "page_budget", "outcome", "missing_operation", "repeated_operation", "source_postimage", "restored_source",
    "generation_reused", "owner_digest", "owner_replay", "owner_cleanup", "unrecorded_response", "wire_count",
    "prompt", "argv_binary", "argv_client", "argv_pin", "provider_tokens", "proof_claim", "catalog_filter",
    "catalog_generation", "source_content", "duplicate_json", "edit_occurrences", "source_summary"])
def test_tampered_or_unmatched_records_refuse_even_when_worker_hash_is_resealed(change):
    value = fixture()
    trace, replay = value["treatment"], value["treatment_replay"]
    call = trace["calls"][0]
    if change == "stream_hash": call["stdout_sha256"] = "0" * 64
    elif change == "stream_size": call["stdout_utf8_bytes"] += 1
    elif change == "bool_size": call["stderr_utf8_bytes"] = False
    elif change == "newline": streams(call, stdout=call["stdout"].rstrip("\n"))
    elif change == "raw_response": call["argv"].append("--raw-response")
    elif change == "extra_field": call["discount"] = 1
    elif change == "page_budget": call["argv"][5] = "4096"
    elif change == "outcome": call["returncode"] = 2
    elif change == "missing_operation": trace["calls"].pop()
    elif change == "repeated_operation": trace["calls"].append(deepcopy(call))
    elif change == "source_postimage": trace["edited_source_hashes"] = {"m.py": "0" * 64}
    elif change == "restored_source": trace["restored_source_hashes"] = {"m.py": "0" * 64}
    elif change == "generation_reused": trace["generations"]["restored"] = trace["generations"]["baseline"]
    elif change == "owner_digest": replay["controls_sha256"] = "0" * 64
    elif change == "owner_replay": replay["operation_replays"][0]["sha256"] = "0" * 64
    elif change == "owner_cleanup": replay["cleanup_verified"] = False
    elif change == "unrecorded_response": replay["delivered_responses"] += 1
    elif change == "wire_count": replay["delivered_response_json_bytes"] -= 1
    elif change == "prompt":
        trace["prompt"]["text"] += "extra fact"
        trace["prompt"]["utf8_bytes"] = len(trace["prompt"]["text"].encode())
        trace["prompt"]["sha256"] = replay["prompt_sha256"] = sha(trace["prompt"]["text"].encode())
    elif change == "argv_binary": call["argv"][0] = "/other/python"
    elif change == "argv_client": call["argv"][3] = "/other/worker_source_client.py"
    elif change == "argv_pin":
        target = trace["calls"][2]
        target["argv"][target["argv"].index("--expected-source-sha256") + 1] = "0" * 64
    elif change == "provider_tokens": trace["provider_tokens"] = 0
    elif change == "proof_claim": trace["source_obligations_discharged"] = 1
    elif change in {"catalog_filter", "catalog_generation"}:
        call = trace["calls"][1]
        page = json.loads(call["stdout"])
        if change == "catalog_filter": page["selection"]["path"] = "m.py"
        else: page["workspace_generation"] = "0" * 64
        streams(call, stdout=wire(page).decode() + "\n")
    elif change == "source_content":
        page = json.loads(call["stdout"]); page["content"] = "x" * page["end_offset"]
        streams(call, stdout=wire(page).decode() + "\n")
    elif change == "duplicate_json": streams(call, stdout=call["stdout"].replace('{"baseline', '{"ok":true,"baseline', 1))
    elif change == "edit_occurrences":
        value["controls"]["source_edit"]["occurrences"] = 3
        for arm in ("baseline", "treatment"):
            value[arm + "_replay"]["controls_sha256"] = sha(wire(value["controls"]))
    else: trace["source_reads"] = []
    seal(value)
    with pytest.raises(comparison.TraceComparisonError):
        comparison.compare_traces(**value)


@pytest.mark.parametrize("change", ["truncate", "overlap", "wrong_cursor", "forged_cursor", "generation", "early_exhaustion"])
def test_partial_pagination_cannot_be_reported_as_a_complete_cheaper_read(change):
    value = fixture(paginate=True, binary_slices=True)
    trace = value["baseline"]
    first, second = trace["calls"][:2]
    page = json.loads(second["stdout"])
    if change == "truncate": trace["calls"].pop(1)
    elif change == "overlap": page["offset"] -= 1
    elif change == "wrong_cursor": second["argv"][second["argv"].index("--cursor") + 1] = "{}"
    elif change == "forged_cursor":
        prior = json.loads(first["stdout"]); prior["next_cursor"]["token"] = "z" * 32
        streams(first, stdout=wire(prior).decode() + "\n")
        second["argv"][second["argv"].index("--cursor") + 1] = json.dumps(prior["next_cursor"])
    elif change == "generation": page["workspace_generation"] = trace["generations"]["restored"]
    else:
        prior = json.loads(first["stdout"]); prior["exhausted"] = True; prior["next_cursor"] = None
        streams(first, stdout=wire(prior).decode() + "\n")
    streams(second, stdout=wire(page).decode() + "\n")
    seal(value, "baseline")
    with pytest.raises(comparison.TraceComparisonError):
        comparison.compare_traces(**value)


@pytest.mark.parametrize("change", ["generation_role", "initial_pin_policy"])
def test_identical_source_bytes_do_not_hide_different_generation_or_request_policy(change):
    value = fixture()
    trace = value["treatment"]
    call = trace["calls"][0]
    index = call["argv"].index("--expected-workspace-generation")
    if change == "generation_role":
        # Restored bytes equal baseline, but this is a different phase of work.
        call["argv"][index + 1] = trace["generations"]["restored"]
        page = json.loads(call["stdout"])
        page["workspace_generation"] = trace["generations"]["restored"]
        streams(call, stdout=wire(page).decode() + "\n")
    else:
        del call["argv"][index:index + 2]
    seal(value)
    with pytest.raises(comparison.TraceComparisonError, match="policies differ"):
        comparison.compare_traces(**value)
