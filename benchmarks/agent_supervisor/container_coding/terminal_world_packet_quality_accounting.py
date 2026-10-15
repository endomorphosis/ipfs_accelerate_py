"""Pure paired quality/accounting over independently pinned observed inputs.

The pre-run control/slot declaration and post-run observation inventory are
different inputs. Neither an inventory flag nor supplied CLI receipts proves an
exhaustive campaign/provider call census. This module performs no I/O, dispatch,
tokenization, verification, pricing, or model inference. A verifier hash binds
caller-trusted evidence; a hash does not establish the truth of that evidence.
"""
from __future__ import annotations

import hashlib
import json
import math
import re

DECLARATION_SCHEMA = "paired-world-quality-declaration@1"
INVENTORY_SCHEMA = "paired-world-quality-observation-inventory@1"
ATTEMPT_SCHEMA = "paired-world-quality-attempt@1"
VERIFIER_SCHEMA = "paired-world-quality-verifier@1"
TOOL_SCHEMA = "paired-world-quality-tool-census@1"
REPORT_SCHEMA = "paired-world-quality-accounting@1"
ARMS = ("baseline", "candidate")
MAX_CASES = 32
MAX_ATTEMPTS = 128
MAX_CHECKS = 256
MAX_TOOL_EVENTS = 4096
MAX_INPUT_BYTES = 16 * 1024 * 1024
MAX_RECEIPT_BYTES = 2 * 1024 * 1024
MAX_REPORT_BYTES = 16 * 1024 * 1024
MAX_NODES = 262_144
MAX_DEPTH = 40
MAX_COUNTER = (1 << 63) - 1
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,127}\Z")
_COMMON = frozenset({"control_sha256", "provider", "model", "model_config_sha256",
    "budgets_sha256", "cache_policy_sha256", "cache_state_sha256",
    "execution_policy_sha256", "verifier_policy_sha256"})
_CASE_PINS = ("source_sha256", "task_sha256", "acceptance_sha256", "requirements_sha256")
_CONTROL_FIELDS = _COMMON | set(_CASE_PINS) | {"codec_sha256"}
_FALSE = dict(default_adoption_authorized=False, semantic_proof_verified=False,
    billing_total_verified=False, provider_ingestion_verified=False,
    provider_delivery_verified=False, token_savings_established=False,
    whole_workflow_token_savings_established=False, completion_authority=False,
    raw_content_exported=False, independent_campaign_census_verified=False,
    verifier_truth_established=False)


class QualityAccountingError(ValueError):
    """Malformed, unpinned, changed or over-bound accounting inputs."""


def _need(condition, reason):
    if not condition:
        raise QualityAccountingError(reason)


def _closed(value, fields, reason):
    _need(type(value) is dict and set(value) == set(fields), reason)


def _sha(value, reason, *, nullable=False):
    _need(nullable and value is None or type(value) is str and _SHA.fullmatch(value), reason)


def _id(value, reason):
    _need(type(value) is str and _ID.fullmatch(value), reason)


def _counter(value, reason, *, nullable=False, maximum=MAX_COUNTER):
    _need(nullable and value is None or type(value) is int and 0 <= value <= maximum, reason)


def _marker(value, reason):
    _need(value is None or type(value) is bool, reason)


def _wire(value, maximum=MAX_INPUT_BYTES):
    # Bound traversal/string expansion before serialization or detached copies.
    pending = [(value, 0)]
    nodes = characters = 0
    while pending:
        item, depth = pending.pop()
        nodes += 1
        _need(nodes <= MAX_NODES and depth <= MAX_DEPTH, "json_structure_bound")
        if type(item) is str:
            characters += len(item)
        elif type(item) is dict:
            _need(all(type(k) is str for k in item) and len(item) + len(pending) <= MAX_NODES,
                  "json_structure_bound")
            characters += sum(len(k) for k in item)
            pending.extend((v, depth + 1) for v in item.values())
        elif type(item) is list:
            _need(len(item) + len(pending) <= MAX_NODES, "json_structure_bound")
            pending.extend((v, depth + 1) for v in item)
        elif type(item) is int:
            _need(item.bit_length() <= 64, "json_integer_bound")
        elif type(item) is float:
            # Complete native receipts include elapsed seconds. Measurements
            # remain JSON data; all declared counters/rewards are checked below.
            _need(math.isfinite(item), "noncanonical_json_value")
        else:
            _need(item is None or type(item) is bool, "noncanonical_json_value")
        _need(characters <= maximum, "json_byte_bound")
    try:
        result = json.dumps(value, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (ValueError, TypeError, UnicodeError, RecursionError) as error:
        raise QualityAccountingError("noncanonical_json_value") from error
    _need(len(result) <= maximum, "json_byte_bound")
    return result


def canonical_input_sha256(value):
    """FULL canonical input hash, including any existing internal seals."""
    return hashlib.sha256(_wire(value)).hexdigest()


def _pin(value, expected, reason, maximum=MAX_INPUT_BYTES):
    _sha(expected, "invalid_expected_pin")
    raw = _wire(value, maximum)
    _need(hashlib.sha256(raw).hexdigest() == expected, reason)
    return raw


def quality_accounting_report_sha256(report):
    """FULL canonical report hash; retain this independently for replay."""
    return hashlib.sha256(_wire(report, MAX_REPORT_BYTES)).hexdigest()


def _declaration(value):
    _closed(value, {"schema", "inventory_scope", "experiment_id", "evidence_kind",
        "common_controls", "cases", "execution_order"}, "declaration_fields")
    _need(value["schema"] == DECLARATION_SCHEMA and
        value["inventory_scope"] == "pre_run_control_and_slot_inventory", "declaration_scope")
    _id(value["experiment_id"], "experiment_id")
    _need(value["evidence_kind"] in ("observed_provider_workflow", "authored_fixture"), "evidence_kind")
    common = value["common_controls"]
    _closed(common, _COMMON, "common_control_fields")
    for key in _COMMON:
        (_id if key in ("provider", "model") else _sha)(common[key], "invalid_common_control")
    cases = value["cases"]
    _need(type(cases) is list and 1 <= len(cases) <= MAX_CASES, "case_population_bound")
    case_ids, slots = [], {}
    for case in cases:
        _closed(case, {"case_id", *_CASE_PINS, "required_check_ids", "arms"}, "case_fields")
        _id(case["case_id"], "case_id")
        case_ids.append(case["case_id"])
        for key in _CASE_PINS:
            _sha(case[key], "case_binding")
        checks = case["required_check_ids"]
        _need(type(checks) is list and 1 <= len(checks) <= MAX_CHECKS, "check_population_bound")
        for check in checks:
            _id(check, "check_id")
        _need(len(set(checks)) == len(checks), "duplicate_check_id")
        _closed(case["arms"], ARMS, "arm_population")
        for arm in ARMS:
            entry = case["arms"][arm]
            _closed(entry, {"codec_sha256", "attempt_slots"}, "arm_fields")
            _sha(entry["codec_sha256"], "codec_pin")
            attempts = entry["attempt_slots"]
            _need(type(attempts) is list and 1 <= len(attempts) <= MAX_ATTEMPTS,
                  "attempt_slot_population_bound")
            previous = set()
            for slot in attempts:
                _closed(slot, {"attempt_id", "parent_attempt_id"}, "attempt_slot_fields")
                identity, parent = slot["attempt_id"], slot["parent_attempt_id"]
                _id(identity, "attempt_id")
                _need(identity not in slots, "duplicate_attempt_id")
                _need(parent is None or type(parent) is str and parent in previous,
                      "retry_parent_not_prior_same_arm")
                # Independent planning/coding calls may each be roots. A
                # non-null parent alone declares a retry of an earlier slot.
                slots[identity] = (case, arm, slot)
                previous.add(identity)
                _need(len(slots) <= MAX_ATTEMPTS, "attempt_slot_population_bound")
    _need(len(set(case_ids)) == len(case_ids), "duplicate_case_id")
    order = value["execution_order"]
    _need(type(order) is list and len(order) == len(slots) and
        all(type(v) is str for v in order) and set(order) == set(slots), "execution_order_population")
    positions = {identity: i for i, identity in enumerate(order)}
    for identity, (_, _, slot) in slots.items():
        parent = slot["parent_attempt_id"]
        _need(parent is None or positions[parent] < positions[identity], "retry_execution_order")
    return slots


def _inventory(value, declaration):
    _closed(value, {"schema", "inventory_scope", "attempt_pins", "verifier_pins",
        "supplied_attempt_census_complete", "unreceipted_provider_attempt"}, "inventory_fields")
    _need(value["schema"] == INVENTORY_SCHEMA and
        value["inventory_scope"] == "post_run_observation_inventory", "inventory_scope")
    for key in ("supplied_attempt_census_complete", "unreceipted_provider_attempt"):
        _marker(value[key], "invalid_census_marker")
    pins = value["attempt_pins"]
    _need(type(pins) is list and len(pins) == len(declaration["execution_order"]), "attempt_pin_population")
    for identity, pin in zip(declaration["execution_order"], pins):
        _closed(pin, {"attempt_id", "attempt_sha256", "receipt_sha256", "tool_trace_sha256"}, "attempt_pin_fields")
        _need(pin["attempt_id"] == identity, "attempt_pin_order")
        _sha(pin["attempt_sha256"], "attempt_pin")
        _sha(pin["receipt_sha256"], "receipt_pin", nullable=True)
        _sha(pin["tool_trace_sha256"], "tool_trace_pin", nullable=True)
    verifiers = value["verifier_pins"]
    expected = [(case["case_id"], arm) for case in declaration["cases"] for arm in ARMS]
    _need(type(verifiers) is list and len(verifiers) == len(expected), "verifier_pin_population")
    for (case_id, arm), pin in zip(expected, verifiers):
        _closed(pin, {"case_id", "arm", "verifier_sha256", "output_sha256", "evaluated_attempt_id"}, "verifier_pin_fields")
        _need((pin["case_id"], pin["arm"]) == (case_id, arm), "verifier_pin_order")
        _sha(pin["verifier_sha256"], "verifier_pin")
        _sha(pin["output_sha256"], "output_pin", nullable=True)
        if pin["evaluated_attempt_id"] is not None:
            _id(pin["evaluated_attempt_id"], "evaluated_attempt_id")
    return pins, verifiers


def _tool_census(value):
    if value is None:
        return dict(status="unavailable", coverage="unknown", event_count=None,
            declared_event_count=None, payload_bytes=None, known_payload_byte_subtotal=None,
            missing_fingerprint_count=None)
    _closed(value, {"schema", "coverage", "declared_event_count", "events"}, "tool_census_fields")
    _need(value["schema"] == TOOL_SCHEMA and value["coverage"] in
        ("complete_declared", "partial", "unknown"), "tool_census_scope")
    _counter(value["declared_event_count"], "tool_declared_count", nullable=True, maximum=MAX_TOOL_EVENTS)
    events = value["events"]
    _need(type(events) is list and len(events) <= MAX_TOOL_EVENTS, "tool_event_bound")
    ids, bytes_known, missing = set(), [], 0
    for event in events:
        _closed(event, {"event_id_sha256", "kind", "bytes", "payload_sha256"}, "tool_event_fields")
        _sha(event["event_id_sha256"], "tool_event_id")
        _need(event["event_id_sha256"] not in ids, "duplicate_tool_event")
        ids.add(event["event_id_sha256"])
        _need(event["kind"] in ("tool_call", "tool_response", "retrieval_request", "retrieval_response"),
              "tool_event_kind")
        _counter(event["bytes"], "tool_bytes", nullable=True)
        _sha(event["payload_sha256"], "tool_payload_pin", nullable=True)
        if event["bytes"] is not None:
            bytes_known.append(event["bytes"])
        if event["payload_sha256"] is None:
            missing += 1
    subtotal = sum(bytes_known) if bytes_known else (0 if not events else None)
    _counter(subtotal, "aggregate_tool_bytes", nullable=True)
    if value["coverage"] == "complete_declared":
        _need(value["declared_event_count"] == len(events), "tool_census_population_mismatch")
    complete_bytes = value["coverage"] == "complete_declared" and len(bytes_known) == len(events)
    return dict(status="observed", coverage=value["coverage"], event_count=len(events),
        declared_event_count=value["declared_event_count"],
        payload_bytes=subtotal if complete_bytes else None, known_payload_byte_subtotal=subtotal,
        missing_fingerprint_count=missing)


def _verifier(value, pin, case, arm, declaration, attempts_by_id):
    _closed(value, {"schema", "case_id", "arm", *_CASE_PINS, "verifier_policy_sha256",
        "provenance", "execution_status", "output_sha256", "evaluated_attempt_id", "checks", "reward"}, "verifier_fields")
    _need(value["schema"] == VERIFIER_SCHEMA and (value["case_id"], value["arm"])
        == (case["case_id"], arm), "verifier_identity")
    _pin(value, pin["verifier_sha256"], "verifier_hash_mismatch")
    _sha(value["output_sha256"], "verifier_output_pin", nullable=True)
    _need(value["output_sha256"] == pin["output_sha256"], "verifier_output_mismatch")
    evaluated = value["evaluated_attempt_id"]
    if evaluated is not None:
        _id(evaluated, "evaluated_attempt_id")
    _need(evaluated == pin["evaluated_attempt_id"], "verifier_attempt_pin_mismatch")
    _need(value["provenance"] in ("independently_retained_verifier", "authored_fixture", "unknown"),
          "verifier_provenance")
    _need(value["execution_status"] in ("completed", "failed", "not_run", "unknown"), "verifier_execution")
    blockers = []
    attempt = attempts_by_id.get(evaluated)
    if attempt is None or attempt["case_id"] != case["case_id"] or attempt["arm"] != arm:
        blockers.append("evaluated_attempt_missing_or_foreign")
    elif attempt["dispatch_state"] != "dispatched":
        blockers.append("evaluated_attempt_not_dispatched")
    elif value["output_sha256"] is None or attempt["output_sha256"] is None:
        blockers.append("evaluated_output_identity_missing")
    elif value["output_sha256"] != attempt["output_sha256"]:
        blockers.append("evaluated_output_identity_mismatch")
    for key in _CASE_PINS:
        _sha(value[key], "verifier_case_binding")
        if value[key] != case[key]:
            blockers.append("verifier_" + key + "_mismatch")
    _sha(value["verifier_policy_sha256"], "verifier_policy_pin")
    if value["verifier_policy_sha256"] != declaration["common_controls"]["verifier_policy_sha256"]:
        blockers.append("verifier_policy_mismatch")
    checks = value["checks"]
    _need(type(checks) is list and len(checks) <= MAX_CHECKS, "verifier_check_bound")
    ids = []
    for check in checks:
        _closed(check, {"check_id", "outcome", "evidence_sha256"}, "check_fields")
        _id(check["check_id"], "check_id")
        ids.append(check["check_id"])
        _need(check["outcome"] in ("pass", "fail", "unknown"), "check_outcome")
        _sha(check["evidence_sha256"], "check_evidence_pin", nullable=True)
    _need(len(set(ids)) == len(ids), "duplicate_verifier_check")
    if ids != case["required_check_ids"]:
        blockers.append("required_check_population_incomplete_or_reordered")
    if any(c["outcome"] == "unknown" or c["evidence_sha256"] is None for c in checks):
        blockers.append("required_check_evidence_incomplete")
    reward = value["reward"]
    _closed(reward, {"outcome", "value", "evidence_sha256"}, "reward_fields")
    _need(reward["outcome"] in ("pass", "fail", "unknown"), "reward_outcome")
    expected_reward = {"pass": 1, "fail": 0, "unknown": None}[reward["outcome"]]
    _need((type(reward["value"]) is int and reward["value"] == expected_reward)
        if expected_reward is not None else reward["value"] is None, "reward_value_mismatch")
    _sha(reward["evidence_sha256"], "reward_evidence_pin", nullable=True)
    if reward["outcome"] == "unknown" or reward["evidence_sha256"] is None:
        blockers.append("reward_evidence_incomplete")
    if value["execution_status"] != "completed":
        blockers.append("verifier_not_completed")
    expected_provenance = ("authored_fixture" if declaration["evidence_kind"] == "authored_fixture"
                           else "independently_retained_verifier")
    if value["provenance"] != expected_provenance:
        blockers.append("verifier_provenance_unqualified")
    if reward["outcome"] == "pass" and (value["output_sha256"] is None or
            any(c["outcome"] == "fail" for c in checks)):
        blockers.append("passing_reward_output_or_check_conflict")
    complete = not blockers
    passed = (reward["outcome"] == "pass" and all(c["outcome"] == "pass" for c in checks)) if complete else None
    return dict(case_id=case["case_id"], arm=arm, verifier_sha256=pin["verifier_sha256"],
        output_sha256=value["output_sha256"], output_present=value["output_sha256"] is not None,
        evaluated_attempt_id=evaluated,
        provenance=value["provenance"], execution_status=value["execution_status"],
        required_check_count=len(case["required_check_ids"]), observed_check_count=len(checks),
        checks=json.loads(_wire(checks)), reward=dict(reward), complete=complete,
        passed=passed, blockers=sorted(set(blockers)))


def compare_paired_quality_usage(declaration, observation_inventory, attempts, verifier_results, *,
        expected_declaration_sha256, expected_observation_inventory_sha256,
        expected_attempts_sha256, expected_verifier_results_sha256):
    """Compare exact pre-run slots and independently pinned post-run evidence.

    ``receipt_sha256`` in the post-run inventory is the FULL raw receipt hash,
    distinct from the core's supported-field projection hash. Native totals
    summarize once per arm, across cases and retries. Supplied-census markers
    condition this declared-workflow comparison; they never certify the absence
    of omitted campaign/auxiliary calls. No report constitutes default adoption.
    """
    # All aggregate pins/bounds precede snapshots, owner projection or retention.
    snapshots = []
    aggregate_bytes = 0
    for value, expected, reason in (
        (declaration, expected_declaration_sha256, "declaration_hash_mismatch"),
        (observation_inventory, expected_observation_inventory_sha256, "inventory_hash_mismatch"),
        (attempts, expected_attempts_sha256, "attempt_population_hash_mismatch"),
        (verifier_results, expected_verifier_results_sha256, "verifier_population_hash_mismatch")):
        raw = _pin(value, expected, reason)
        aggregate_bytes += len(raw)
        _need(aggregate_bytes <= MAX_INPUT_BYTES, "aggregate_input_byte_bound")
        snapshots.append(raw)
    declaration, observation_inventory, attempts, verifier_results = (
        json.loads(raw) for raw in snapshots)
    slots = _declaration(declaration)
    attempt_pins, verifier_pins = _inventory(observation_inventory, declaration)
    _need(type(attempts) is list and len(attempts) == len(slots), "attempt_population_mismatch")
    _need(type(verifier_results) is list and len(verifier_results) == 2 * len(declaration["cases"]),
          "verifier_population_mismatch")
    from ipfs_accelerate_py.agent_supervisor.runtime import context_delivery_accounting as core

    arm_observations = {arm: [] for arm in ARMS}
    attempt_rows, structural = [], []
    marker = observation_inventory["unreceipted_provider_attempt"]
    missing_by_arm = dict.fromkeys(ARMS, False)
    tool_events = 0
    invocation_slots = set()
    for identity, value, pin in zip(declaration["execution_order"], attempts, attempt_pins):
        _closed(value, {"schema", "attempt_id", "case_id", "arm", "parent_attempt_id",
            "dispatch_state", "outcome", "dispatch_evidence_sha256", "observed_controls",
            "receipt", "tool_trace", "output_sha256"}, "attempt_fields")
        _pin(value, pin["attempt_sha256"], "attempt_hash_mismatch")
        case, arm, slot = slots[identity]
        _need(value["schema"] == ATTEMPT_SCHEMA and value["attempt_id"] == identity and
            (value["case_id"], value["arm"], value["parent_attempt_id"]) ==
            (case["case_id"], arm, slot["parent_attempt_id"]), "attempt_slot_identity_mismatch")
        state, outcome = value["dispatch_state"], value["outcome"]
        _need(state in ("dispatched", "refused", "unknown", "not_dispatched"), "dispatch_state")
        _need(outcome in ("succeeded", "failed", "refused", "unknown", "not_attempted"), "attempt_outcome")
        _need((state == "refused" and outcome == "refused") or
            (state == "not_dispatched" and outcome == "not_attempted") or
            (state == "dispatched" and outcome in ("succeeded", "failed", "unknown")) or
            (state == "unknown" and outcome == "unknown"), "dispatch_outcome_conflict")
        _sha(value["dispatch_evidence_sha256"], "dispatch_evidence_pin", nullable=True)
        _sha(value["output_sha256"], "attempt_output_pin", nullable=True)
        blockers = []
        if state in ("unknown",) or value["dispatch_evidence_sha256"] is None:
            blockers.append("dispatch_evidence_incomplete")
        controls = value["observed_controls"]
        _closed(controls, _CONTROL_FIELDS, "observed_control_fields")
        expected_controls = {**declaration["common_controls"], **{k: case[k] for k in _CASE_PINS},
            "codec_sha256": case["arms"][arm]["codec_sha256"]}
        for key, expected in expected_controls.items():
            if key in ("provider", "model"):
                if controls[key] is not None:
                    _id(controls[key], "observed_route")
            else:
                _sha(controls[key], "observed_control_pin", nullable=True)
            if controls[key] != expected:
                blockers.append("observed_" + key + "_unmatched")
        receipt = value["receipt"]
        _need((receipt is None) == (pin["receipt_sha256"] is None), "receipt_pin_presence_mismatch")
        observed = None
        if receipt is not None:
            _need(type(receipt) is dict, "receipt_fields")
            _pin(receipt, pin["receipt_sha256"], "receipt_hash_mismatch", MAX_RECEIPT_BYTES)
            projection = {key: v for key, v in receipt.items() if key in core.OBSERVE_INPUT_FIELDS}
            observed = core.observe_invocation(projection)
            arm_observations[arm].append(observed)
            if observed["invocation_id_sha256"] in invocation_slots:
                blockers.append("invocation_reused_across_attempt_slots")
            invocation_slots.add(observed["invocation_id_sha256"])
            if observed["provider"] != expected_controls["provider"] or observed["model"] != expected_controls["model"]:
                blockers.append("receipt_provider_model_mismatch")
            if state in ("refused", "not_dispatched"):
                blockers.append("non_dispatch_slot_has_provider_receipt")
        if ((state in ("dispatched", "unknown") and observed is None) or state == "unknown" or
                value["dispatch_evidence_sha256"] is None):
            missing_by_arm[arm] = True
        tool = value["tool_trace"]
        _need((tool is None) == (pin["tool_trace_sha256"] is None), "tool_pin_presence_mismatch")
        if tool is not None:
            _pin(tool, pin["tool_trace_sha256"], "tool_trace_hash_mismatch")
        tool_row = _tool_census(tool)
        tool_events += tool_row["event_count"] or 0
        _need(tool_events <= MAX_TOOL_EVENTS, "aggregate_tool_event_bound")
        structural.extend(blockers)
        attempt_rows.append(dict(attempt_id=identity, case_id=case["case_id"], arm=arm,
            parent_attempt_id=slot["parent_attempt_id"], dispatch_state=state, outcome=outcome,
            attempt_sha256=pin["attempt_sha256"], full_receipt_sha256=pin["receipt_sha256"],
            output_sha256=value["output_sha256"],
            dispatch_evidence_sha256=value["dispatch_evidence_sha256"],
            invocation_id_sha256=observed["invocation_id_sha256"] if observed else None,
            observation_sha256=observed["observation_sha256"] if observed else None,
            tool_trace_sha256=pin["tool_trace_sha256"], tool_census=tool_row, blockers=sorted(set(blockers))))

    # Same invocation/session used by both treatments cannot be independently charged.
    def identities(arm):
        return {(row["provider"], row["native_usage"]["session_id_sha256"])
            for row in arm_observations[arm] if row["native_usage"]["session_id_sha256"] is not None}
    shared_sessions = identities("baseline") & identities("candidate")
    shared_invocations = ({r["invocation_id_sha256"] for r in arm_observations["baseline"]} &
                          {r["invocation_id_sha256"] for r in arm_observations["candidate"]})
    if shared_sessions:
        structural.append("cross_arm_native_session_overlap")
    if shared_invocations:
        structural.append("cross_arm_invocation_overlap")
    summaries = {}
    for arm in ARMS:
        arm_marker = True if marker is True or missing_by_arm[arm] else marker
        if observation_inventory["supplied_attempt_census_complete"] is not True and arm_marker is False:
            arm_marker = None
        summaries[arm] = core.summarize_invocations(arm_observations[arm],
            unreceipted_provider_attempt=arm_marker)

    quality_rows = []
    attempts_by_id = {row["attempt_id"]: row for row in attempt_rows}
    offset = 0
    for case in declaration["cases"]:
        pair = {}
        for arm in ARMS:
            pair[arm] = _verifier(verifier_results[offset], verifier_pins[offset], case, arm,
                declaration, attempts_by_id)
            offset += 1
        known = all(pair[arm]["complete"] for arm in ARMS)
        nonworse = (int(pair["candidate"]["passed"]) >= int(pair["baseline"]["passed"])) if known else None
        quality_rows.append(dict(case_id=case["case_id"], arms=pair, complete=known, candidate_nonworse=nonworse))
    structural_ok = not structural
    quality_complete = all(row["complete"] for row in quality_rows)
    nonworse = all(row["candidate_nonworse"] is True for row in quality_rows) if quality_complete else None
    usage_complete = structural_ok and all(summaries[a]["complete_supplied_native_usage"] for a in ARMS)
    totals = {arm: summaries[arm]["complete_native_total_tokens"] if usage_complete else None for arm in ARMS}
    lower = totals["candidate"] < totals["baseline"] if usage_complete else None
    scientific_blockers = sorted(set(structural +
        ([] if quality_complete else ["quality_evidence_incomplete"]) +
        ([] if nonworse is True else ["candidate_quality_regressed_or_unknown"]) +
        ([] if usage_complete else ["declared_native_usage_incomplete"]) +
        ([] if lower is True else ["complete_native_usage_not_strictly_lower"])))
    criteria_met = not scientific_blockers
    actual = declaration["evidence_kind"] == "observed_provider_workflow"
    result = dict(schema=REPORT_SCHEMA, experiment_id=declaration["experiment_id"],
        evidence_kind=declaration["evidence_kind"], observation_only=True,
        input_pins=dict(declaration_sha256=expected_declaration_sha256,
            observation_inventory_sha256=expected_observation_inventory_sha256,
            attempts_sha256=expected_attempts_sha256, verifier_results_sha256=expected_verifier_results_sha256),
        population=dict(case_count=len(quality_rows), arm_count=2, attempt_slot_count=len(attempt_rows),
            dispatched_attempt_count=sum(r["dispatch_state"] == "dispatched" for r in attempt_rows),
            refused_attempt_count=sum(r["dispatch_state"] == "refused" for r in attempt_rows),
            unknown_attempt_count=sum(r["dispatch_state"] == "unknown" for r in attempt_rows),
            not_attempted_slot_count=sum(r["dispatch_state"] == "not_dispatched" for r in attempt_rows),
            failed_dispatched_attempt_count=sum(r["outcome"] == "failed" for r in attempt_rows),
            retry_slot_count=sum(r["parent_attempt_id"] is not None for r in attempt_rows),
            receipt_count=sum(r["full_receipt_sha256"] is not None for r in attempt_rows)),
        attempts=attempt_rows, native_usage_by_arm=summaries, quality_by_case=quality_rows,
        structural_gate=dict(qualified=structural_ok, blockers=sorted(set(structural))),
        quality_gate=dict(complete=quality_complete, candidate_nonworse=nonworse),
        usage_gate=dict(complete=usage_complete, complete_native_total_tokens=totals,
            strictly_lower_candidate_native_usage=lower),
        scientific_gate=dict(comparison_scope="declared_observed_workflow_usage_comparison",
            criteria_met=criteria_met, observed_efficiency_qualified=criteria_met and actual,
            blockers=scientific_blockers + ([] if actual else ["authored_fixture_not_observed_provider_workflow"])),
        tool_trace_scope=dict(scope="supplied_declared_tool_or_retrieval_events",
            observed_event_count=tool_events, provider_token_attribution=None,
            event_count_scope="event observations across supplied per-slot censuses",
            cross_attempt_event_identity_deduplicated=False,
            tool_bytes_added_to_native_tokens=False, model_ingestion_verified=False),
        coverage=dict(pre_run_control_and_slot_pin_verified=True,
            post_run_observation_pins_verified=True,
            supplied_attempt_census_complete=observation_inventory["supplied_attempt_census_complete"],
            unreceipted_provider_attempt=marker, campaign_call_population_complete=None,
            provider_auxiliary_call_population_complete=None, provider_internal_retries_known=None,
            native_usage_scope="supplied CLI sessions or final envelopes, selected once per arm",
            quality_scope="independently pinned supplied verifier checks and output identities",
            control_temporal_origin_verified=False, verifier_authentication_verified=False), **_FALSE)
    _wire(result, MAX_REPORT_BYTES)
    return result


def validate_quality_accounting_report(report, declaration, observation_inventory, attempts,
        verifier_results, *, expected_report_sha256, **trusted_pins):
    """Validate an independent FULL report pin and replay all original inputs."""
    _pin(report, expected_report_sha256, "report_hash_mismatch", MAX_REPORT_BYTES)
    regenerated = compare_paired_quality_usage(declaration, observation_inventory, attempts,
        verifier_results, **trusted_pins)
    _need(_wire(report, MAX_REPORT_BYTES) == _wire(regenerated, MAX_REPORT_BYTES), "report_replay_mismatch")
    return dict(schema="paired-world-quality-accounting-validation@1", valid=True,
        report_sha256=expected_report_sha256, case_count=regenerated["population"]["case_count"],
        attempt_slot_count=regenerated["population"]["attempt_slot_count"], **_FALSE)
