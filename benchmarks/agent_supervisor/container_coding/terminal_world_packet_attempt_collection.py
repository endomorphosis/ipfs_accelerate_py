"""Pure collection of an exact declared population of durable attempt records.

All input and component hashes are independently supplied observations. A hash
binds supplied bytes; it does not authenticate a journal, a verifier, chronology,
provider execution, output contents, or an exhaustive campaign. Missing records
remain explicit unknown attempts. This module does not read files or stdout,
invoke callbacks or providers, run verifiers, or reconstruct native usage.
"""
from __future__ import annotations

import hashlib
import json
import re

from benchmarks.agent_supervisor.container_coding import terminal_world_packet_quality_accounting as accounting

CENSUS_SCHEMA = "paired-world-attempt-slot-census@1"
COLLECTION_SCHEMA = "paired-world-attempt-collection@1"
COMPARISON_SCHEMA = "paired-world-collected-quality-comparison@1"
START_SCHEMA = "repository-world-attempt-start@1"
TERMINAL_SCHEMA = "repository-world-attempt-terminal@1"
START_MAX_BYTES = TERMINAL_MAX_BYTES = 64 * 1024
_CONTROLS = accounting._CONTROL_FIELDS
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:/@+\-]{0,127}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_ERROR = re.compile(r"[A-Za-z_][A-Za-z0-9_]{0,127}\Z")
_FALSE = dict(default_adoption_authorized=False, provider_execution_authenticated=False,
    journal_authentication_verified=False, output_contents_verified=False,
    verifier_truth_established=False, chronology_verified=False,
    control_temporal_origin_verified=False, independent_campaign_census_verified=False,
    provider_ingestion_verified=False, billing_total_verified=False,
    whole_workflow_token_savings_established=False, completion_authority=False)


class AttemptCollectionError(ValueError):
    """Malformed, unpinned, conflicting, foreign or over-bound supplied records."""


def _need(value, code):
    if not value:
        raise AttemptCollectionError(code)


def _closed(value, fields, code):
    _need(type(value) is dict and set(value) == set(fields), code)


def _sha(value, code, *, nullable=False):
    _need(nullable and value is None or type(value) is str and _SHA.fullmatch(value), code)


def _id(value, code, *, nullable=False):
    _need(nullable and value is None or type(value) is str and _ID.fullmatch(value), code)


def _wire(value, maximum=accounting.MAX_INPUT_BYTES):
    # Share the downstream finite JSON, depth, node and pre-serialization bounds.
    try:
        return accounting._wire(value, maximum)
    except accounting.QualityAccountingError as error:
        raise AttemptCollectionError(str(error)) from error


def _pin(value):
    return hashlib.sha256(_wire(value)).hexdigest()


def _snapshot(value, expected, code, maximum=accounting.MAX_INPUT_BYTES):
    _sha(expected, "invalid_expected_pin")
    raw = _wire(value, maximum)
    _need(hashlib.sha256(raw).hexdigest() == expected, code)
    return raw


def attempt_collection_sha256(value):
    """FULL canonical collection/comparison pin, including any internal seals."""
    return _pin(value)


def _component(slot, name, maximum):
    value, pin = slot[name], slot[name + "_sha256"]
    _need((value is None) == (pin is None), name + "_pin_presence_mismatch")
    if value is not None:
        _need(type(value) is dict, name + "_fields")
        _snapshot(value, pin, name + "_hash_mismatch", maximum)
    return value


def _start(value, identity, case, arm, declared_slot):
    if value is None:
        return dict.fromkeys(_CONTROLS)
    _closed(value, {"schema", "attempt_id", "case_id", "arm", "parent_attempt_id",
        "observed_controls"}, "start_fields")
    _need(value["schema"] == START_SCHEMA and
        (value["attempt_id"], value["case_id"], value["arm"], value["parent_attempt_id"]) ==
        (identity, case["case_id"], arm, declared_slot["parent_attempt_id"]), "start_slot_identity_mismatch")
    controls = value["observed_controls"]
    _closed(controls, _CONTROLS, "start_observed_control_fields")
    for key, observed in controls.items():
        (_id if key in ("provider", "model") else _sha)(observed,
            "invalid_observed_control", nullable=True)
    return dict(controls)


def _receipt(value):
    if value is None:
        return "unknown"
    _need(value.get("schema") == "router-implementation-invocation@1" and
        value.get("provider") == "codex_cli" and value.get("purpose") == "coding" and
        type(value.get("invocation_id")) is str and
        re.fullmatch(r"[0-9a-f]{32}", value["invocation_id"]) is not None, "receipt_schema_or_route")
    _id(value.get("model"), "receipt_model")
    _need(value.get("status") in ("provider_returned", "failed"), "receipt_status")
    _need(value.get("completion_authority") is False, "receipt_completion_authority")
    _id(value.get("error_type"), "receipt_error_type", nullable=True)
    packet = value.get("world_task_packet")
    _need(type(packet) is dict and "validation_profile" not in packet and
        type(packet.get("validation_count")) is int and packet["validation_count"] > 0 and
        type(packet.get("dispatch_rechecked")) is bool, "receipt_strict_packet_observation")
    marker = packet.get("provider_attempted")
    _need(type(marker) is bool, "receipt_provider_attempted")
    _need(not marker or packet["dispatch_rechecked"], "receipt_dispatch_not_rechecked")
    if marker:
        return "dispatched"
    _need(value.get("status") != "provider_returned", "receipt_dispatch_status_conflict")
    return "refused"


def _terminal(value, identity, start, start_pin, receipt, receipt_pin):
    if value is None:
        return
    _closed(value, {"schema", "attempt_id", "start_sha256", "receipt_sha256",
        "outcome", "error_type"}, "terminal_fields")
    _need(value["schema"] == TERMINAL_SCHEMA and value["attempt_id"] == identity,
        "terminal_slot_identity_mismatch")
    _sha(value["start_sha256"], "terminal_start_pin")
    _sha(value["receipt_sha256"], "terminal_receipt_pin", nullable=True)
    _need(value["outcome"] in ("returned", "raised", "interrupted"), "terminal_outcome")
    _need(value["error_type"] is None or type(value["error_type"]) is str and
        _ERROR.fullmatch(value["error_type"]), "terminal_error_type")
    _need(value["outcome"] != "returned" or value["error_type"] is None, "terminal_returned_error")
    if start is not None:
        _need(value["start_sha256"] == start_pin, "terminal_start_join_mismatch")
    if receipt is not None:
        _need(value["receipt_sha256"] == receipt_pin, "terminal_receipt_join_mismatch")


def _unknown_verifiers(declaration):
    return [dict(schema=accounting.VERIFIER_SCHEMA, case_id=case["case_id"], arm=arm,
        **{key: case[key] for key in accounting._CASE_PINS},
        verifier_policy_sha256=declaration["common_controls"]["verifier_policy_sha256"],
        provenance="unknown", execution_status="unknown", output_sha256=None,
        evaluated_attempt_id=None,
        checks=[dict(check_id=identity, outcome="unknown", evidence_sha256=None)
            for identity in case["required_check_ids"]],
        reward=dict(outcome="unknown", value=None, evidence_sha256=None))
        for case in declaration["cases"] for arm in accounting.ARMS]


def collect_paired_attempt_observations(declaration, slot_census, verifier_results=None, *,
        expected_declaration_sha256, expected_slot_census_sha256,
        expected_verifier_results_sha256=None):
    """Collect every predeclared slot without filling missing observed evidence.

    ``slot_census`` is a caller-retained, independently FULL-pinned ordered
    inventory of decoded journal components and their independent canonical
    pins. ``output_sha256`` is an explicitly supplied retained-output identity;
    neither a provider return nor its answer text supplies this observation.
    Existing amended declarations are accepted only as supplied pinned inputs;
    no amendment order or temporal origin is authenticated here.

    Derived accounting inputs receive reproducible derived pins. These are
    clearly separate from the independently supplied original input pins.
    """
    raw_declaration = _snapshot(declaration, expected_declaration_sha256,
        "declaration_hash_mismatch")
    raw_census = _snapshot(slot_census, expected_slot_census_sha256, "slot_census_hash_mismatch")
    aggregate_bytes = len(raw_declaration) + len(raw_census)
    if verifier_results is None:
        _need(expected_verifier_results_sha256 is None, "unexpected_verifier_pin")
        raw_verifiers = None
    else:
        raw_verifiers = _snapshot(verifier_results, expected_verifier_results_sha256,
            "verifier_population_hash_mismatch")
        aggregate_bytes += len(raw_verifiers)
    _need(aggregate_bytes <= accounting.MAX_INPUT_BYTES, "aggregate_input_byte_bound")
    declaration, census = json.loads(raw_declaration), json.loads(raw_census)
    try:
        slots = accounting._declaration(declaration)
    except accounting.QualityAccountingError as error:
        raise AttemptCollectionError(str(error)) from error
    _closed(census, {"schema", "inventory_scope", "supplied_attempt_census_complete",
        "unreceipted_provider_attempt", "slots"}, "slot_census_fields")
    _need(census["schema"] == CENSUS_SCHEMA and
        census["inventory_scope"] == "supplied_declared_journal_records", "slot_census_scope")
    for key in ("supplied_attempt_census_complete", "unreceipted_provider_attempt"):
        _need(census[key] is None or type(census[key]) is bool, "invalid_census_marker")
    records = census["slots"]
    _need(type(records) is list and len(records) == len(slots), "slot_record_population")
    attempts, collected, all_blockers = [], [], []
    for identity, record in zip(declaration["execution_order"], records):
        _closed(record, {"attempt_id", "start", "start_sha256", "receipt", "receipt_sha256",
            "terminal", "terminal_sha256", "output_sha256"}, "slot_record_fields")
        _need(record["attempt_id"] == identity, "slot_record_order_or_identity")
        case, arm, declared_slot = slots[identity]
        start = _component(record, "start", START_MAX_BYTES)
        receipt = _component(record, "receipt", accounting.MAX_RECEIPT_BYTES)
        terminal = _component(record, "terminal", TERMINAL_MAX_BYTES)
        _sha(record["output_sha256"], "output_pin", nullable=True)
        controls = _start(start, identity, case, arm, declared_slot)
        state = _receipt(receipt)
        _terminal(terminal, identity, start, record["start_sha256"], receipt, record["receipt_sha256"])
        if state == "refused":
            outcome = "refused"
        elif state == "unknown":
            outcome = "unknown"
        elif receipt.get("status") == "failed" or terminal is not None and terminal["outcome"] != "returned":
            outcome = "failed"
        elif terminal is not None and receipt.get("status") == "provider_returned":
            outcome = "succeeded"
        else:
            outcome = "unknown"
        blockers = [name + "_missing" for name, value in
            (("start", start), ("receipt", receipt), ("terminal", terminal)) if value is None]
        if start is None:
            blockers.append("observed_controls_missing")
        if terminal is not None and start is None:
            blockers.append("terminal_start_join_unavailable")
        if terminal is not None and receipt is None and terminal["receipt_sha256"] is not None:
            blockers.append("terminal_receipt_join_unavailable")
        if record["output_sha256"] is None:
            blockers.append("output_identity_missing")
        if state == "unknown":
            blockers.append("dispatch_unknown")
        if outcome == "unknown":
            blockers.append("attempt_outcome_unknown")
        all_blockers.extend(blockers)
        attempts.append(dict(schema=accounting.ATTEMPT_SCHEMA, attempt_id=identity,
            case_id=case["case_id"], arm=arm, parent_attempt_id=declared_slot["parent_attempt_id"],
            dispatch_state=state, outcome=outcome,
            dispatch_evidence_sha256=record["receipt_sha256"] if receipt is not None else None,
            observed_controls=controls, receipt=receipt, tool_trace=None,
            output_sha256=record["output_sha256"]))
        collected.append(dict(attempt_id=identity, start_sha256=record["start_sha256"],
            receipt_sha256=record["receipt_sha256"], terminal_sha256=record["terminal_sha256"],
            output_sha256=record["output_sha256"],
            recorded_terminal_outcome=terminal["outcome"] if terminal is not None else None,
            recorded_error_type=terminal["error_type"] if terminal is not None else None,
            blockers=sorted(set(blockers))))
    verifiers = json.loads(raw_verifiers) if raw_verifiers is not None else _unknown_verifiers(declaration)
    _need(type(verifiers) is list and len(verifiers) == 2 * len(declaration["cases"]),
        "verifier_population_mismatch")
    inventory = dict(schema=accounting.INVENTORY_SCHEMA, inventory_scope="post_run_observation_inventory",
        supplied_attempt_census_complete=census["supplied_attempt_census_complete"],
        unreceipted_provider_attempt=census["unreceipted_provider_attempt"],
        attempt_pins=[dict(attempt_id=row["attempt_id"], attempt_sha256=_pin(row),
            receipt_sha256=_pin(row["receipt"]) if row["receipt"] is not None else None,
            tool_trace_sha256=None) for row in attempts],
        verifier_pins=[dict(case_id=row.get("case_id"), arm=row.get("arm"),
            verifier_sha256=_pin(row), output_sha256=row.get("output_sha256"),
            evaluated_attempt_id=row.get("evaluated_attempt_id"))
            for row in verifiers if type(row) is dict])
    _need(len(inventory["verifier_pins"]) == len(verifiers), "verifier_fields")
    accounting_pins = dict(zip(("expected_declaration_sha256", "expected_observation_inventory_sha256",
        "expected_attempts_sha256", "expected_verifier_results_sha256"),
        map(_pin, (declaration, inventory, attempts, verifiers))))
    # The same closed validators used by consumers reject malformed supplied
    # receipts/verifiers before a collected population can be published.
    try:
        accounting.compare_paired_quality_usage(declaration, inventory, attempts, verifiers, **accounting_pins)
    except accounting.QualityAccountingError as error:
        raise AttemptCollectionError(str(error)) from error
    result = dict(schema=COLLECTION_SCHEMA, observation_only=True,
        input_pins=dict(declaration_sha256=expected_declaration_sha256,
            slot_census_sha256=expected_slot_census_sha256,
            verifier_results_sha256=expected_verifier_results_sha256),
        declaration=declaration, attempts=attempts, observation_inventory=inventory,
        verifier_results=verifiers, accounting_input_pins=accounting_pins,
        accounting_pin_origin="derived_from_independently_pinned_supplied_inputs",
        collected_slots=collected, blockers=sorted(set(all_blockers)),
        durable_population_gate=dict(qualified=not any(code != "output_identity_missing" for code in all_blockers),
            blockers=sorted(set(code for code in all_blockers if code != "output_identity_missing")),
            scope="supplied_declared_start_receipt_terminal_records_and_recorded_outcomes",
            generated_output_required_for_lifecycle=False),
        collection_coverage=dict(declared_slot_count=len(slots), collected_slot_count=len(attempts),
            missing_start_count=sum(row["start_sha256"] is None for row in collected),
            missing_receipt_count=sum(row["receipt_sha256"] is None for row in collected),
            missing_terminal_count=sum(row["terminal_sha256"] is None for row in collected),
            output_identity_scope="explicitly_supplied_retained_output_digest",
            output_identity_missing_count=sum(row["output_sha256"] is None for row in collected),
            supplied_attempt_census_complete=census["supplied_attempt_census_complete"],
            unreceipted_provider_attempt=census["unreceipted_provider_attempt"],
            campaign_call_population_complete=None, provider_internal_retries_known=None,
            tool_trace_collection="not_supplied", stdout_parsed=False,
            verifier_origin="independently_pinned_supplied_population" if raw_verifiers is not None else "generated_unknown_population"),
        **_FALSE)
    _wire(result)
    return result


def compare_collected_paired_quality_usage(declaration, slot_census, verifier_results=None, **pins):
    """Compose collection and the existing session-safe paired comparator."""
    collection = collect_paired_attempt_observations(declaration, slot_census, verifier_results, **pins)
    report = accounting.compare_paired_quality_usage(collection["declaration"],
        collection["observation_inventory"], collection["attempts"], collection["verifier_results"],
        **collection["accounting_input_pins"])
    result = dict(schema=COMPARISON_SCHEMA, collection=collection,
        collection_sha256=attempt_collection_sha256(collection), accounting_report=report,
        accounting_report_sha256=accounting.quality_accounting_report_sha256(report),
        durable_population_gate=dict(collection["durable_population_gate"]),
        comparison_gate=dict(comparison_scope="declared_workflow_accounting_and_durable_record_coverage",
            criteria_met=collection["durable_population_gate"]["qualified"] and report["scientific_gate"]["criteria_met"],
            observed_efficiency_qualified=collection["durable_population_gate"]["qualified"] and
                report["scientific_gate"]["observed_efficiency_qualified"],
            blockers=sorted(set(["durable_" + code for code in collection["durable_population_gate"]["blockers"]]
                + report["scientific_gate"]["blockers"]))), **_FALSE)
    _wire(result)
    return result
