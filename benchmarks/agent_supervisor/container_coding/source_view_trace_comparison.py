"""Compare complete scripted read/edit traces, without claiming provider usage.

The caller supplies retained CLI text and independent owner replay receipts.
Receipt hashes bind those records; they do not authenticate the owner, prove
provider delivery, or turn a scripted trace into an official task result.
No stores, model calls, executable source transforms or context compiler live
here. The existing worker client's default rendering remains the measured API.
"""
from __future__ import annotations

import base64
import hashlib
import json
from pathlib import PurePosixPath

SCHEMA = "source-view-trace-comparison@1"
CONTROLS_SCHEMA = "source-view-trace-controls@1"
REPLAY_SCHEMA = "source-view-trace-owner-replay@1"
TRACE_SCHEMA = "authored-source-read-edit-trace@1"
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_CALLS = 64
MAX_TRACE_BYTES = 16 * 1024 * 1024


class TraceComparisonError(ValueError):
    """Captured traces do not establish the same complete scripted workload."""


def _need(condition, reason):
    if not condition:
        raise TraceComparisonError(reason)


def canonical_json(value):
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise TraceComparisonError("canonical finite JSON required") from exc


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _fields(value, required, optional=()):
    _need(type(value) is dict and set(required) <= set(value)
          and set(value) <= set(required) | set(optional), "closed record fields differ")


def _integer(value, low=0, high=MAX_TRACE_BYTES):
    return type(value) is int and low <= value <= high


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _text(value, *, empty=True):
    _need(type(value) is str, "exact UTF-8 text required")
    raw = value.encode("utf-8")
    _need((empty or bool(raw)) and len(raw) <= MAX_TEXT_BYTES, "text byte bound exceeded")
    return raw


def _path(value):
    _text(value, empty=False)
    _need(len(value.encode()) <= 4096 and not PurePosixPath(value).is_absolute()
          and str(PurePosixPath(value)) == value and value != "."
          and not any(p in {"..", ".git"} for p in PurePosixPath(value).parts)
          and "\\" not in value and all(ord(c) >= 32 and ord(c) != 127 for c in value),
          "canonical admitted relative path required")


def _hashes(value, *, paths=False):
    _need(type(value) is dict and 0 < len(value) <= 256, "bounded hash inventory required")
    for name, digest in value.items():
        _text(name, empty=False)
        if paths:
            _path(name)
        _need(_digest(digest), "exact SHA-256 inventory required")


def _loads(text):
    def pairs(items):
        result = {}
        for key, value in items:
            _need(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(text, object_pairs_hook=pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(TraceComparisonError("nonfinite JSON")))


def _controls(value):
    _fields(value, {"schema", "common_input_sha256", "runtime_sha256", "script_sha256",
                    "page_budget", "source_hashes", "logical_operations", "source_edit"})
    _need(value["schema"] == CONTROLS_SCHEMA, "unsupported controls schema")
    _hashes(value["common_input_sha256"])
    _need("native_prompt" in value["common_input_sha256"], "original owner prompt pin required")
    _hashes(value["runtime_sha256"])
    _hashes(value["source_hashes"], paths=True)
    _need(_digest(value["script_sha256"]) and _integer(value["page_budget"], 2048, 65536),
          "script and page budget controls required")
    operations = value["logical_operations"]
    _need(type(operations) is list and 1 <= len(operations) <= MAX_CALLS, "bounded logical trace required")
    seen = set()
    for operation in operations:
        _fields(operation, {"operation_id", "kind", "path", "view", "outcome"})
        _text(operation["operation_id"], empty=False)
        _need(operation["operation_id"] not in seen and operation["kind"] in {"source", "catalog"}
              and operation["outcome"] in {"complete", "refused"}, "logical operation identity differs")
        seen.add(operation["operation_id"])
        if operation["kind"] == "source":
            _need(operation["path"] in value["source_hashes"]
                  and operation["view"] in {"compact", "original"}, "source operation outside admitted scope")
        else:
            _need(operation["path"] is None and operation["view"] is None,
                  "catalog operation has source selection")
    edit = value["source_edit"]
    _fields(edit, {"path", "old", "new", "occurrences", "before_sha256", "after_sha256"})
    _need(edit["path"] in value["source_hashes"] and edit["before_sha256"] == value["source_hashes"][edit["path"]]
          and _digest(edit["after_sha256"]) and edit["before_sha256"] != edit["after_sha256"]
          and _integer(edit["occurrences"], 1, 65536), "exact original-source edit controls required")
    _text(edit["old"], empty=False)
    _text(edit["new"])
    _need(edit["old"] != edit["new"], "nontrivial edit required")
    return value


def _argv(call, controls):
    argv = call["argv"]
    _need(type(argv) is list and 4 <= len(argv) <= 64, "bounded captured argv required")
    for item in argv:
        _text(item, empty=False)
        _need("\0" not in item, "NUL in argv")
    client = "worker_source_client.py" if call["kind"] == "source" else "worker_candidate_context_client.py"
    _need(argv[1:3] == ["-I", "-B"] and PurePosixPath(argv[3]).name == client,
          "fixed isolated default client required")
    switches = {"--working", "--select"}
    valued = {"--max-bytes", "--path", "--view", "--cursor", "--expected-workspace-generation",
              "--expected-source-sha256", "--expected-view-sha256", "--artifact"}
    options, position = {}, 4
    while position < len(argv):
        flag = argv[position]
        _need(flag not in options and flag in switches | valued, "unexpected or duplicate CLI flag")
        if flag in switches:
            options[flag] = True
            position += 1
        else:
            _need(position + 1 < len(argv), "missing CLI option value")
            options[flag] = argv[position + 1]
            position += 2
    _need(options.get("--max-bytes") == str(controls["page_budget"]), "CLI page budget differs")
    if call["kind"] == "source":
        _need(options.get("--path") == call["path"] and options.get("--view") == call["view"]
              and options.get("--working") is True and "--select" not in options
              and "--artifact" not in options, "source CLI selection differs")
        if call["view"] == "original" or "--cursor" in options:
            _need(all(flag in options for flag in ("--expected-source-sha256", "--expected-view-sha256",
                                                  "--expected-workspace-generation")),
                  "exact expansion and continuation pins required")
    else:
        _need(options.get("--select") is True and options.get("--artifact") == "catalog"
              and not set(options) & {"--path", "--view", "--working", "--expected-source-sha256", "--expected-view-sha256"},
              "catalog CLI selection differs")
    for key in ("--expected-workspace-generation", "--expected-source-sha256", "--expected-view-sha256"):
        _need(key not in options or _digest(options[key]), "invalid CLI generation/source pin")
    _need(("--expected-source-sha256" in options) == ("--expected-view-sha256" in options),
          "paired expansion pins required")
    return options


def _payload(page):
    _need(page["encoding"] in {"utf-8", "base64"}, "unsupported displayed slice encoding")
    raw = _text(page["content"])
    if page["encoding"] == "base64":
        raw = base64.b64decode(raw, validate=True)
        _need(base64.b64encode(raw).decode() == page["content"], "noncanonical base64")
    return raw


def _operation(calls, operation, controls, generations, source_states):
    pieces, previous, identities = [], None, None
    for index, call in enumerate(calls):
        _need(call["page_index"] == index, "page sequence differs")
        options = _argv(call, controls)
        page = _loads(call["stdout"])
        _need(type(page) is dict and type(page.get("ok")) is bool, "default CLI result required")
        _need(call["stdout"].encode() == canonical_json(page) + b"\n", "exact default CLI JSON and newline required")
        if operation["outcome"] == "refused":
            _fields(page, {"ok", "reason_code"})
            _need(len(calls) == 1 and page["ok"] is False and call["returncode"] == 2,
                  "expected refusal was not observed")
            _need(page["reason_code"] in {"invalid_request", "peer_refused", "stale_context",
                  "budget_exhausted", "session_closed", "retrieval_refused"}, "unknown source refusal")
            return None
        _need(page["ok"] is True and call["returncode"] == 0, "successful complete read required")
        fields = {"ok", "offset", "end_offset", "exhausted", "next_cursor", "encoding", "content",
                  "source_state", "workspace_generation", "baseline_evidence_applicable"}
        source = operation["kind"] == "source"
        fields |= ({"path", "view", "source_sha256", "view_sha256", "baseline_source_sha256"} if source else
                   {"artifact", "artifact_sha256", "artifact_bytes", "proof_authority", "completion_authority",
                    "selection", "source_obligations_discharged"})
        _fields(page, fields)
        generation = page["workspace_generation"]
        _need(generation in generations.values() and page["source_state"] == "working-candidate"
              and page["baseline_evidence_applicable"] is False, "candidate generation or labels differ")
        expected_generation = options.get("--expected-workspace-generation")
        _need(expected_generation is None or expected_generation == generation, "CLI generation differs from page")
        if source:
            _need(page["path"] == operation["path"] and page["view"] == operation["view"],
                  "displayed source selection differs")
            state = next(name for name, value in generations.items() if value == generation)
            _need(page["baseline_source_sha256"] == controls["source_hashes"][page["path"]]
                  and page["source_sha256"] == source_states[state][page["path"]]
                  and _digest(page["view_sha256"]), "displayed source identity differs")
            for key in ("source_sha256", "view_sha256"):
                pin = options.get("--expected-" + key.replace("_", "-"))
                _need(pin is None or pin == page[key], "expansion pin differs from source")
            identity = (generation, page["source_sha256"], page["view_sha256"])
            digest = page["source_sha256"] if page["view"] == "original" else page["view_sha256"]
        else:
            _need(page["artifact"] == "catalog" and _digest(page["artifact_sha256"])
                  and _integer(page["artifact_bytes"]) and page["proof_authority"] is False
                  and page["completion_authority"] is False and type(page["source_obligations_discharged"]) is int
                  and page["source_obligations_discharged"] == 0
                  and page["selection"] == dict(path=None, symbol_id=None, function_id=None, kind=None),
                  "catalog identity, filters or nonproof labels differ")
            _need(expected_generation == generation, "catalog generation pin required")
            identity = (generation, page["artifact_sha256"], page["artifact_bytes"])
            digest = page["artifact_sha256"]
        _need(identities is None or identities == identity, "identity changed within paginated read")
        identities = identity
        _need(_integer(page["offset"]) and _integer(page["end_offset"])
              and page["offset"] == sum(map(len, pieces)), "missing, repeated or overlapping slice")
        raw = _payload(page)
        _need(page["end_offset"] == page["offset"] + len(raw), "displayed slice size differs")
        pieces.append(raw)
        cursor = options.get("--cursor")
        _need((cursor is None and previous is None) or (cursor is not None and _loads(cursor) == previous),
              "CLI continuation differs from preceding page")
        _need(type(page["exhausted"]) is bool and page["exhausted"] == (index == len(calls) - 1),
              "trace ended before exhaustion or continued after completion")
        previous = page["next_cursor"]
        if page["exhausted"]:
            _need(previous is None, "exhausted page retains continuation")
        else:
            _fields(previous, {"schema", "request_sha256", "offset", "token"})
            expected_schema = ("supervisor-worker-source-view-cursor@1" if source else
                               "supervisor-worker-candidate-context-cursor@1")
            _need(previous["schema"] == expected_schema and _integer(previous["offset"])
                  and previous["offset"] == page["end_offset"] and raw and _digest(previous["request_sha256"])
                  and type(previous["token"]) is str and len(previous["token"]) == 32
                  and all(c in "0123456789abcdef" for c in previous["token"]),
                  "invalid progressing continuation")
    assembled = b"".join(pieces)
    _need(_sha(assembled) == digest, "assembled representation digest differs")
    if not source:
        _need(len(assembled) == page["artifact_bytes"], "catalog incomplete")
        _need(type(_loads(assembled.decode())) is dict, "catalog must decode to an object")
    return dict(operation_id=operation["operation_id"], sha256=digest, bytes=len(assembled)), assembled


def _arm(trace, replay, controls):
    required = {"schema", "profile", "calls", "prompt", "source_hashes", "edited_source_hashes",
                "restored_source_hashes", "generations", "provider_calls", "model_calls", "provider_tokens",
                "source_obligations_discharged", "total_token_savings_measured", "edits_mapped",
                "actual_original_source_edit", "original_source_restored", "full_program_behavior_verified"}
    optional = {"uid", "gid", "workspace", "descriptor_path", "descriptor_sha256", "private_read_denials",
                "no_new_privileges", "worker_effective_capabilities", "limitations", "source_reads", "source_edit", "page_budget", "task_alias",
                "catalog_step_optional", "input_paths_known_from_manifest_and_fixture"}
    _fields(trace, required, optional)
    _need(trace["schema"] == TRACE_SCHEMA and trace["profile"] in {"identity@1", "python-whitespace@1", "python-comments@1"},
          "unsupported captured trace")
    for field in ("provider_calls", "model_calls", "source_obligations_discharged"):
        _need(type(trace[field]) is int and trace[field] == 0, "scripted census has no provider/proof calls")
    _need(trace["provider_tokens"] is None and all(trace[k] is False for k in
          ("total_token_savings_measured", "edits_mapped", "full_program_behavior_verified"))
          and trace["actual_original_source_edit"] is True and trace["original_source_restored"] is True,
          "trace scope or authority differs")
    raw_trace = canonical_json(trace)
    _need(len(raw_trace) <= MAX_TRACE_BYTES, "trace byte bound exceeded")
    _fields(replay, {"schema", "trace_sha256", "controls_sha256", "prompt_sha256", "operation_replays",
        "source_hashes", "edited_source_hashes", "restored_source_hashes", "response_json_bytes",
        "delivered_response_json_bytes", "delivered_responses", "accounting_complete", "source_replay_verified",
        "source_edit_verified", "cleanup_verified"})
    _need(replay["schema"] == REPLAY_SCHEMA and replay["trace_sha256"] == _sha(raw_trace)
          and replay["controls_sha256"] == _sha(canonical_json(controls))
          and all(replay[k] is True for k in ("accounting_complete", "source_replay_verified", "source_edit_verified", "cleanup_verified")),
          "owner replay binding or completion differs")
    edit = controls["source_edit"]
    _need("page_budget" not in trace or (type(trace["page_budget"]) is int and trace["page_budget"] == controls["page_budget"]),
          "captured page budget differs")
    if "source_edit" in trace:
        _need(trace["source_edit"] == dict(operation="replace_all_exact_bytes", path=edit["path"],
              preimage=edit["old"], replacement=edit["new"], expected_occurrences=edit["occurrences"],
              before_sha256=edit["before_sha256"], after_sha256=edit["after_sha256"]), "captured edit description differs")
    edited = dict(controls["source_hashes"], **{edit["path"]: edit["after_sha256"]})
    for key, expected in (("source_hashes", controls["source_hashes"]), ("edited_source_hashes", edited),
                          ("restored_source_hashes", controls["source_hashes"])):
        _need(trace[key] == expected == replay[key], "source postimage or owner replay differs")
    generations = trace["generations"]
    _fields(generations, {"baseline", "edited", "restored"})
    _need(all(_digest(g) for g in generations.values()) and len(set(generations.values())) == 3,
          "distinct baseline/edit/restore generations required")
    prompt = trace["prompt"]
    _fields(prompt, {"text", "utf8_bytes", "sha256"})
    raw_prompt = _text(prompt["text"], empty=False)
    _need(type(prompt["utf8_bytes"]) is int and prompt["utf8_bytes"] == len(raw_prompt)
          and prompt["sha256"] == _sha(raw_prompt) == replay["prompt_sha256"], "captured prompt identity differs")
    calls = trace["calls"]
    _need(type(calls) is list and 1 <= len(calls) <= MAX_CALLS, "bounded actual call trace required")
    groups, order, sums = {}, [], dict(stdout_bytes=0, stderr_bytes=0, scripted_argv_json_bytes=0)
    for index, call in enumerate(calls):
        _fields(call, {"index", "operation_id", "page_index", "phase", "kind", "path", "view", "argv", "returncode",
                      "stdout", "stderr", "stdout_utf8_bytes", "stderr_utf8_bytes", "stdout_sha256", "stderr_sha256"})
        _need(type(call["index"]) is int and call["index"] == index and _integer(call["page_index"], 0, MAX_CALLS),
              "call index differs")
        _text(call["operation_id"], empty=False)
        _text(call["phase"], empty=False)
        _need(call["operation_id"] == call["phase"] + ":" + (call["path"] or call["kind"]),
              "operation phase identity differs")
        for stream in ("stdout", "stderr"):
            raw = _text(call[stream])
            _need(type(call[stream + "_utf8_bytes"]) is int and call[stream + "_utf8_bytes"] == len(raw)
                  and call[stream + "_sha256"] == _sha(raw), "captured CLI stream identity differs")
            sums[stream + "_bytes"] += len(raw)
        _need(_integer(call["returncode"], 0, 255) and len(call["stdout"].encode()) <= controls["page_budget"] + 1,
              "CLI result exceeds bounds")
        sums["scripted_argv_json_bytes"] += len(canonical_json(call["argv"]))
        name = call["operation_id"]
        if name not in groups:
            groups[name] = []
            order.append(name)
        _need(order[-1] == name, "interleaved or repeated logical operation")
        groups[name].append(call)
    _need(order == [op["operation_id"] for op in controls["logical_operations"]], "logical operation sequence differs")
    replays, originals, source_reads, refusals, operation_controls = [], {}, [], [], []
    states = dict(baseline=trace["source_hashes"], edited=trace["edited_source_hashes"], restored=trace["restored_source_hashes"])
    for operation in controls["logical_operations"]:
        group = groups[operation["operation_id"]]
        _need(all(all(row[k] == operation[k] for k in ("kind", "path", "view")) for row in group), "operation selection differs")
        result = _operation(group, operation, controls, generations, states)
        first_options = _argv(group[0], controls)
        requested = first_options.get("--expected-workspace-generation")
        _need(requested is None or requested in generations.values(), "request uses unknown captured generation")
        first_page = _loads(group[0]["stdout"])
        def role(generation):
            return next((name for name, value in generations.items() if value == generation), None)
        operation_controls.append(dict(operation_id=operation["operation_id"],
            requested_generation_role=role(requested), returned_generation_role=role(first_page.get("workspace_generation")),
            initial_source_pins="--expected-source-sha256" in first_options))
        if result is not None:
            replay_row, assembled = result
            replays.append(replay_row)
            if operation["kind"] == "source" and operation["view"] == "original" and operation["path"] == edit["path"]:
                originals[_sha(assembled)] = assembled
            if operation["kind"] == "source":
                first = _loads(group[0]["stdout"])
                source_reads.append(dict(operation_id=operation["operation_id"], phase=group[0]["phase"],
                    path=operation["path"], view=operation["view"], assembled_sha256=replay_row["sha256"],
                    assembled_bytes=replay_row["bytes"], page_count=len(group),
                    **{key: first[key] for key in ("source_sha256", "view_sha256", "workspace_generation")}))
        else:
            refusals.append(dict(operation_id=operation["operation_id"], reason_code=_loads(group[0]["stdout"])["reason_code"]))
    _need(type(replay["operation_replays"]) is list, "owner operation replay list required")
    for row in replay["operation_replays"]:
        _fields(row, {"operation_id", "sha256", "bytes"})
        _need(_digest(row["sha256"]) and _integer(row["bytes"]), "typed owner replay identity required")
    _need(replays == replay["operation_replays"], "independent owner representation replay differs")
    _need("source_reads" not in trace or canonical_json(trace["source_reads"]) == canonical_json(source_reads),
          "captured source read summary differs from actual output")
    _need(edit["before_sha256"] in originals and edit["after_sha256"] in originals, "both exact edit pre/postimage expansions required")
    before, after = originals[edit["before_sha256"]], originals[edit["after_sha256"]]
    _need(before.count(edit["old"].encode()) == edit["occurrences"]
          and before.replace(edit["old"].encode(), edit["new"].encode()) == after, "prescribed original-source edit differs")
    for key in ("response_json_bytes", "delivered_response_json_bytes", "delivered_responses"):
        _need(_integer(replay[key]), "integer transport accounting required")
    _need(replay["delivered_responses"] == len(calls)
          and replay["response_json_bytes"] == replay["delivered_response_json_bytes"]
          and sums["stdout_bytes"] - len(calls) <= replay["delivered_response_json_bytes"] <= controls["page_budget"] * len(calls),
          "transport accounting is incomplete or inconsistent")
    return dict(profile=trace["profile"], trace_sha256=_sha(raw_trace), prompt_sha256=prompt["sha256"],
        prepared_prompt_bytes=len(raw_prompt), **sums, calls=len(calls), logical_operations=len(order),
        scripted_exposure_bytes=len(raw_prompt) + sums["stdout_bytes"] + sums["stderr_bytes"],
        transport_response_json_bytes=replay["response_json_bytes"],
        transport_delivered_response_json_bytes=replay["delivered_response_json_bytes"],
        transport_delivered_responses=replay["delivered_responses"], complete_operations=replays, refusals=refusals,
        operation_controls=operation_controls)


def compare_traces(*, baseline, treatment, controls, baseline_replay, treatment_replay):
    """Validate a matched identity/compact trace and return content-free byte totals.

    Owner replay records are caller-supplied evidence, not authenticated handles.
    ``scripted_exposure_bytes`` counts prepared input and captured default CLI
    stdout/stderr once. Real providers may resend history at every turn; neither
    those repeated inputs nor model output, billing or official reward is inferred.
    Request-side argv uses canonical JSON, not an invented provider tool wrapper.
    """
    try:
        controls = _controls(controls)
        _need(baseline["profile"] == "identity@1" and treatment["profile"] in {"python-whitespace@1", "python-comments@1"},
              "identity baseline and explicit compact treatment required")
        left = _arm(baseline, baseline_replay, controls)
        right = _arm(treatment, treatment_replay, controls)
        from ipfs_accelerate_py.agent_supervisor.runtime.worker_source_client import guidance
        prompts = []
        for trace in (baseline, treatment):
            advice = guidance(trace["profile"])
            _need(trace["prompt"]["text"].count(advice) == 1, "exact existing source guidance required once")
            prompts.append(trace["prompt"]["text"].replace(advice, ""))
        _need(prompts[0] == prompts[1], "prepared input changed beyond source-view guidance")
        for kind in {row["kind"] for row in baseline["calls"]}:
            clients = {tuple(row["argv"][:4]) for trace in (baseline, treatment) for row in trace["calls"] if row["kind"] == kind}
            _need(len(clients) == 1, "client interpreter/path controls differ")
        _need(left["refusals"] == right["refusals"], "refusal outcomes differ across arms")
        _need(left["operation_controls"] == right["operation_controls"], "generation or initial-pin policies differ across arms")
        for key in ("workspace", "uid", "gid", "task_alias"):
            _need(baseline.get(key) == treatment.get(key), "worker execution controls differ")
        return dict(schema=SCHEMA, controls_sha256=_sha(canonical_json(controls)), baseline=left, treatment=right,
            baseline_minus_treatment_scripted_exposure_bytes=left["scripted_exposure_bytes"] - right["scripted_exposure_bytes"],
            scope="one-pass prepared prompt plus captured default CLI stdout/stderr; scripted argv and transport JSON separate",
            provider_tokens=None, official_reward=None, provider_received_input=False, provider_calls=0, model_calls=0,
            total_token_savings_measured=False, owner_receipts_authenticated=False, source_obligations_discharged=0,
            proof_authority=False, completion_authority=False, mapped_edits_compared=False)
    except TraceComparisonError:
        raise
    except (KeyError, TypeError, ValueError, UnicodeError, RecursionError) as exc:
        raise TraceComparisonError("invalid captured trace") from exc


__all__ = ["compare_traces", "canonical_json", "TraceComparisonError", "SCHEMA", "CONTROLS_SCHEMA", "REPLAY_SCHEMA"]
