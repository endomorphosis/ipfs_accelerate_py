"""Authored journal populations exercise collection, never live model evidence."""
from copy import deepcopy
import hashlib
import json

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_world_packet_attempt_collection as owner
from benchmarks.agent_supervisor.container_coding import terminal_world_packet_quality_accounting as accounting
from benchmarks.agent_supervisor.container_coding.test_terminal_world_packet_quality_accounting import _bundle


def _pin(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _population():
    declaration, _, attempts, verifiers = _bundle()
    records = []
    for attempt in attempts:
        start = {key: deepcopy(attempt[key]) for key in
            ("attempt_id", "case_id", "arm", "parent_attempt_id", "observed_controls")}
        start["schema"] = owner.START_SCHEMA
        receipt = deepcopy(attempt["receipt"])
        receipt["completion_authority"] = False
        receipt["invocation_id"] = hashlib.sha256(attempt["attempt_id"].encode()).hexdigest()[:32]
        receipt["world_task_packet"] = dict(provider_attempted=True, dispatch_rechecked=True, validation_count=1)
        terminal = dict(schema=owner.TERMINAL_SCHEMA, attempt_id=attempt["attempt_id"],
            start_sha256=_pin(start), receipt_sha256=_pin(receipt),
            outcome="returned", error_type=None)
        records.append(dict(attempt_id=attempt["attempt_id"], start=start, start_sha256=_pin(start),
            receipt=receipt, receipt_sha256=_pin(receipt), terminal=terminal,
            terminal_sha256=_pin(terminal), output_sha256=attempt["output_sha256"]))
    census = dict(schema=owner.CENSUS_SCHEMA, inventory_scope="supplied_declared_journal_records",
        supplied_attempt_census_complete=True, unreceipted_provider_attempt=False, slots=records)
    return declaration, census, verifiers


def _pins(declaration, census, verifiers=None):
    return dict(expected_declaration_sha256=_pin(declaration), expected_slot_census_sha256=_pin(census),
        expected_verifier_results_sha256=_pin(verifiers) if verifiers is not None else None)


def _collect(declaration, census, verifiers=None):
    return owner.collect_paired_attempt_observations(declaration, census, verifiers,
        **_pins(declaration, census, verifiers))


def _compare(declaration, census, verifiers=None):
    return owner.compare_collected_paired_quality_usage(declaration, census, verifiers,
        **_pins(declaration, census, verifiers))


def _reseal(record, name):
    record[name + "_sha256"] = _pin(record[name]) if record[name] is not None else None


def _reseal_join(record):
    _reseal(record, "start")
    _reseal(record, "receipt")
    if record["terminal"] is not None:
        record["terminal"]["start_sha256"] = record["start_sha256"]
        record["terminal"]["receipt_sha256"] = record["receipt_sha256"]
        _reseal(record, "terminal")


def test_defaults_keep_unknown_verifiers_and_exact_slot_population():
    declaration, census, _ = _population()
    result = _compare(declaration, census)
    collection, report = result["collection"], result["accounting_report"]
    assert len(collection["attempts"]) == 2
    assert [row["attempt_id"] for row in collection["attempts"]] == declaration["execution_order"]
    assert report["quality_gate"]["complete"] is False
    assert report["scientific_gate"]["criteria_met"] is False
    assert all(row["output_sha256"] is None and row["evaluated_attempt_id"] is None
        and row["execution_status"] == "unknown" for row in collection["verifier_results"])
    assert collection["collection_coverage"]["verifier_origin"] == "generated_unknown_population"


def test_fully_pinned_authored_population_composes_existing_session_safe_comparator():
    declaration, census, verifiers = _population()
    result = _compare(declaration, census, verifiers)
    report = result["accounting_report"]
    assert report["quality_gate"] == dict(complete=True, candidate_nonworse=True)
    assert report["usage_gate"]["complete_native_total_tokens"] == dict(baseline=120, candidate=80)
    assert report["scientific_gate"]["criteria_met"] is True
    assert report["scientific_gate"]["observed_efficiency_qualified"] is False
    assert result["collection_sha256"] == owner.attempt_collection_sha256(result["collection"])
    assert result["accounting_report_sha256"] == accounting.quality_accounting_report_sha256(report)
    assert all(result[key] is False for key in owner._FALSE)
    assert all(result["collection"][key] is False for key in owner._FALSE)


def test_full_receipt_unknown_observations_and_nulls_are_retained_without_input_aliases():
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["receipt"]["unknown_failure_observations"] = [{"message": "literal", "counter": None}, {"message": "literal", "counter": None}]
    _reseal_join(record)
    before = deepcopy((declaration, census))
    result = _collect(declaration, census)
    assert (declaration, census) == before
    retained = result["attempts"][0]["receipt"]
    assert retained == record["receipt"]
    record["receipt"]["unknown_failure_observations"].clear()
    assert len(retained["unknown_failure_observations"]) == 2


def test_receipt_survives_missing_start_without_invented_observed_controls():
    declaration, census, verifiers = _population()
    record = census["slots"][0]
    record["start"] = record["start_sha256"] = None
    result = _compare(declaration, census, verifiers)
    attempt = result["collection"]["attempts"][0]
    assert attempt["receipt"] == record["receipt"]
    assert all(value is None for value in attempt["observed_controls"].values())
    assert result["accounting_report"]["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] == 120
    assert result["accounting_report"]["usage_gate"]["complete"] is False
    assert "terminal_start_join_unavailable" in result["collection"]["blockers"]


def test_receipt_survives_missing_terminal_but_provider_return_does_not_close_wrapper():
    declaration, census, verifiers = _population()
    census["slots"][0]["terminal"] = census["slots"][0]["terminal_sha256"] = None
    result = _compare(declaration, census, verifiers)
    attempt = result["collection"]["attempts"][0]
    assert attempt["dispatch_state"] == "dispatched" and attempt["outcome"] == "unknown"
    assert result["accounting_report"]["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] == 120
    assert "terminal_missing" in result["collection"]["blockers"]


def test_terminal_can_record_missing_receipt_without_fabricating_zero_usage():
    declaration, census, verifiers = _population()
    census["slots"][0]["receipt"] = census["slots"][0]["receipt_sha256"] = None
    result = _compare(declaration, census, verifiers)
    attempt = result["collection"]["attempts"][0]
    assert attempt["dispatch_state"] == "unknown" and attempt["outcome"] == "unknown"
    assert result["accounting_report"]["usage_gate"]["complete_native_total_tokens"] == dict(baseline=None, candidate=None)
    assert "terminal_receipt_join_unavailable" in result["collection"]["blockers"]


def test_all_records_missing_still_has_both_slots_no_native_zero_no_nondispatch_inference():
    declaration, census, _ = _population()
    for row in census["slots"]:
        for key in row:
            if key != "attempt_id": row[key] = None
    census["supplied_attempt_census_complete"] = None
    census["unreceipted_provider_attempt"] = None
    result = _compare(declaration, census)
    assert len(result["collection"]["attempts"]) == 2
    assert all(row["dispatch_state"] == "unknown" and row["outcome"] == "unknown"
        and row["receipt"] is None and row["output_sha256"] is None
        for row in result["collection"]["attempts"])
    assert result["accounting_report"]["usage_gate"]["complete"] is False
    assert result["accounting_report"]["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] is None
    assert result["collection"]["observation_inventory"]["unreceipted_provider_attempt"] is None


@pytest.mark.parametrize("terminal_outcome", ["raised", "interrupted"])
def test_postprovider_wrapper_failure_retains_usage_and_marks_attempt_failed(terminal_outcome):
    declaration, census, verifiers = _population()
    record = census["slots"][0]
    record["terminal"].update(outcome=terminal_outcome, error_type="RuntimeError")
    _reseal(record, "terminal")
    result = _compare(declaration, census, verifiers)
    assert result["collection"]["attempts"][0]["outcome"] == "failed"
    assert result["accounting_report"]["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] == 120
    assert result["collection"]["collected_slots"][0]["recorded_error_type"] == "RuntimeError"


def test_provider_failure_without_terminal_retains_failed_record_and_native_cost():
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["receipt"].update(status="failed", error_type="RateLimitError")
    record["terminal"] = None
    _reseal(record, "receipt")
    _reseal(record, "terminal")
    result = _compare(declaration, census)
    assert result["collection"]["attempts"][0]["outcome"] == "failed"
    assert result["accounting_report"]["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] == 120


def test_provider_return_never_supplies_generated_output_identity():
    declaration, census, verifiers = _population()
    census["slots"][1]["output_sha256"] = None
    result = _compare(declaration, census, verifiers)
    assert result["collection"]["attempts"][1]["outcome"] == "succeeded"
    assert result["collection"]["attempts"][1]["output_sha256"] is None
    assert result["accounting_report"]["quality_gate"]["complete"] is False
    assert result["accounting_report"]["scientific_gate"]["criteria_met"] is False


@pytest.mark.parametrize("marker", [None, False, True])
def test_global_census_marker_is_preserved_as_supplied_never_inferred(marker):
    declaration, census, _ = _population()
    census["unreceipted_provider_attempt"] = marker
    result = _collect(declaration, census)
    assert result["observation_inventory"]["unreceipted_provider_attempt"] is marker
    assert result["collection_coverage"]["campaign_call_population_complete"] is None
    assert result["collection_coverage"]["provider_internal_retries_known"] is None


@pytest.mark.parametrize("target", ["declaration", "census", "verifiers"])
def test_whole_original_input_pins_cannot_be_replaced_by_posthoc_rehash(target):
    declaration, census, verifiers = _population()
    pins = _pins(declaration, census, verifiers)
    if target == "declaration": declaration["experiment_id"] = "foreign"
    elif target == "census": census["unreceipted_provider_attempt"] = None
    else: verifiers[0]["output_sha256"] = "e" * 64
    with pytest.raises(owner.AttemptCollectionError, match="hash_mismatch"):
        owner.collect_paired_attempt_observations(declaration, census, verifiers, **pins)


@pytest.mark.parametrize("component", ["start", "receipt", "terminal"])
def test_full_census_rehash_cannot_replace_independent_component_pin(component):
    declaration, census, _ = _population()
    record = census["slots"][0]
    if component == "start": record[component]["observed_controls"]["source_sha256"] = "e" * 64
    elif component == "receipt": record[component]["unknown"] = "new observation"
    else: record[component]["outcome"] = "raised"
    with pytest.raises(owner.AttemptCollectionError, match=component + "_hash_mismatch"):
        _collect(declaration, census)


@pytest.mark.parametrize("component", ["start", "receipt", "terminal"])
@pytest.mark.parametrize("missing", ["value", "pin"])
def test_present_component_requires_its_independent_pin(component, missing):
    declaration, census, _ = _population()
    census["slots"][0][component if missing == "value" else component + "_sha256"] = None
    with pytest.raises(owner.AttemptCollectionError, match="pin_presence_mismatch"):
        _collect(declaration, census)


@pytest.mark.parametrize("change", ["missing", "extra", "reordered", "duplicate"])
def test_slot_census_cannot_omit_append_reorder_or_duplicate_declared_slots(change):
    declaration, census, _ = _population()
    if change == "missing": census["slots"].pop()
    elif change == "extra": census["slots"].append(deepcopy(census["slots"][0]))
    elif change == "reordered": census["slots"].reverse()
    else: census["slots"][1] = deepcopy(census["slots"][0])
    with pytest.raises(owner.AttemptCollectionError, match="slot_record_"):
        _collect(declaration, census)


@pytest.mark.parametrize("field", ["attempt_id", "case_id", "arm", "parent_attempt_id"])
def test_present_start_cannot_change_declared_slot_identity(field):
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["start"][field] = "foreign"
    _reseal_join(record)
    with pytest.raises(owner.AttemptCollectionError, match="start_slot_identity_mismatch"):
        _collect(declaration, census)


@pytest.mark.parametrize("field", ["start_sha256", "receipt_sha256"])
def test_present_terminal_cannot_join_foreign_component(field):
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["terminal"][field] = "e" * 64
    _reseal(record, "terminal")
    with pytest.raises(owner.AttemptCollectionError, match="terminal_.*_join_mismatch"):
        _collect(declaration, census)


@pytest.mark.parametrize("value", [False, "bad error with spaces", "x/y", "7Error"])
def test_terminal_diagnostic_requires_closed_identifier(value):
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["terminal"].update(outcome="raised", error_type=value)
    _reseal(record, "terminal")
    with pytest.raises(owner.AttemptCollectionError, match="terminal_error_type"):
        _collect(declaration, census)


def test_returned_terminal_cannot_contain_failure_diagnostic():
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["terminal"]["error_type"] = "RuntimeError"
    _reseal(record, "terminal")
    with pytest.raises(owner.AttemptCollectionError, match="terminal_returned_error"):
        _collect(declaration, census)


@pytest.mark.parametrize("field", ["source_sha256", "codec_sha256", "model"])
def test_observed_controls_are_not_filled_from_declaration_when_unknown(field):
    declaration, census, verifiers = _population()
    record = census["slots"][1]
    record["start"]["observed_controls"][field] = None
    _reseal_join(record)
    result = _compare(declaration, census, verifiers)
    assert result["collection"]["attempts"][1]["observed_controls"][field] is None
    assert result["accounting_report"]["usage_gate"]["complete"] is False


@pytest.mark.parametrize("marker", [None, "missing"])
def test_present_receipt_without_strict_dispatch_observation_is_malformed(marker):
    declaration, census, _ = _population()
    record = census["slots"][0]
    if marker is None: record["receipt"]["world_task_packet"] = None
    else: record["receipt"]["world_task_packet"] = {}
    _reseal_join(record)
    with pytest.raises(owner.AttemptCollectionError, match="receipt_strict_packet_observation"):
        _collect(declaration, census)


def test_preprovider_refusal_preserves_actual_receipt_and_is_not_dispatched():
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["receipt"]["world_task_packet"]["provider_attempted"] = False
    record["receipt"].update(status="failed", native_rollout_usage=None)
    record["terminal"].update(outcome="raised", error_type="ValueError")
    _reseal_join(record)
    result = _collect(declaration, census)
    assert result["attempts"][0]["dispatch_state"] == "refused"
    assert result["attempts"][0]["outcome"] == "refused"
    assert result["attempts"][0]["receipt"] == record["receipt"]


@pytest.mark.parametrize("bad", [0, 1, "true"])
def test_dispatch_marker_must_be_exact_boolean(bad):
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["receipt"]["world_task_packet"]["provider_attempted"] = bad
    _reseal_join(record)
    with pytest.raises(owner.AttemptCollectionError, match="receipt_provider_attempted"):
        _collect(declaration, census)


def test_receipt_cannot_claim_successful_provider_return_after_declared_nondispatch():
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["receipt"]["world_task_packet"]["provider_attempted"] = False
    _reseal_join(record)
    with pytest.raises(owner.AttemptCollectionError, match="receipt_dispatch_status_conflict"):
        _collect(declaration, census)


def test_supplied_verifier_must_have_its_external_full_population_pin():
    declaration, census, verifiers = _population()
    pins = _pins(declaration, census)
    with pytest.raises(owner.AttemptCollectionError, match="invalid_expected_pin"):
        owner.collect_paired_attempt_observations(declaration, census, verifiers, **pins)
    pins["expected_verifier_results_sha256"] = _pin(verifiers)
    with pytest.raises(owner.AttemptCollectionError, match="unexpected_verifier_pin"):
        owner.collect_paired_attempt_observations(declaration, census, **pins)


@pytest.mark.parametrize("target", ["census", "slot", "start", "terminal"])
def test_closed_structural_records_reject_unknown_fields(target):
    declaration, census, _ = _population()
    record = census["slots"][0]
    if target == "census": census["unknown"] = True
    elif target == "slot": record["unknown"] = True
    else:
        record[target]["unknown"] = True
        _reseal_join(record)
    with pytest.raises(owner.AttemptCollectionError, match="fields"):
        _collect(declaration, census)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1 << 100])
def test_nonfinite_or_unbounded_unknown_receipt_data_is_not_discarded(value):
    declaration, census, _ = _population()
    census["slots"][0]["receipt"]["unknown"] = value
    with pytest.raises(owner.AttemptCollectionError):
        owner.collect_paired_attempt_observations(declaration, census,
            expected_declaration_sha256=_pin(declaration), expected_slot_census_sha256="e" * 64)


def test_codec_default_does_not_change_global_evidence_kind_or_approvals():
    declaration, census, _ = _population()
    result = _collect(declaration, census)
    assert result["declaration"] == declaration
    assert result["accounting_pin_origin"] == "derived_from_independently_pinned_supplied_inputs"
    assert result["collection_coverage"]["stdout_parsed"] is False
    assert result["collection_coverage"]["tool_trace_collection"] == "not_supplied"
    assert all(result[key] is False for key in owner._FALSE)


@pytest.mark.parametrize("field,value", [
    ("provider", "grok_cli"), ("purpose", "planning"), ("schema", "foreign"),
    ("invocation_id", "wrong"), ("invocation_id", "A" * 32), ("model", None),
    ("model", "bad model"), ("status", None), ("status", "unknown"),
    ("completion_authority", True), ("completion_authority", None),
])
def test_present_receipt_requires_the_supported_exact_journal_route(field, value):
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["receipt"][field] = value
    _reseal_join(record)
    with pytest.raises(owner.AttemptCollectionError, match="receipt_"):
        _collect(declaration, census)


@pytest.mark.parametrize("field,value", [
    ("validation_profile", "read_only_observation@1"), ("validation_profile", None),
    ("validation_count", 0), ("validation_count", True), ("validation_count", None),
    ("dispatch_rechecked", None), ("dispatch_rechecked", 1), ("dispatch_rechecked", False),
])
def test_readonly_or_unchecked_receipt_cannot_be_promoted_into_dispatch(field, value):
    declaration, census, _ = _population()
    record = census["slots"][0]
    record["receipt"]["world_task_packet"][field] = value
    _reseal_join(record)
    with pytest.raises(owner.AttemptCollectionError, match="receipt_"):
        _collect(declaration, census)


def test_missing_terminal_cannot_qualify_durable_comparison_even_if_core_outputs_and_usage_qualify():
    declaration, census, verifiers = _population()
    declaration["evidence_kind"] = "observed_provider_workflow"
    for row in verifiers:
        row["provenance"] = "independently_retained_verifier"
    census["slots"][0]["terminal"] = census["slots"][0]["terminal_sha256"] = None
    result = _compare(declaration, census, verifiers)
    core = result["accounting_report"]
    assert core["scientific_gate"]["observed_efficiency_qualified"] is True
    assert core["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] == 120
    assert result["durable_population_gate"]["qualified"] is False
    assert result["comparison_gate"]["criteria_met"] is False
    assert result["comparison_gate"]["observed_efficiency_qualified"] is False
    assert "durable_terminal_missing" in result["comparison_gate"]["blockers"]
    collection = result["collection"]
    accounting.validate_quality_accounting_report(core, collection["declaration"],
        collection["observation_inventory"], collection["attempts"], collection["verifier_results"],
        expected_report_sha256=result["accounting_report_sha256"], **collection["accounting_input_pins"])


def test_generated_output_is_separate_from_complete_durable_lifecycle():
    declaration, census, verifiers = _population()
    census["slots"][1]["output_sha256"] = None
    result = _compare(declaration, census, verifiers)
    assert result["durable_population_gate"]["qualified"] is True
    assert result["accounting_report"]["quality_gate"]["complete"] is False
    assert result["comparison_gate"]["criteria_met"] is False


@pytest.mark.parametrize("baseline_terminal_retained", [True, False])
def test_real_journal_descriptors_collect_without_inferred_generated_output(tmp_path, baseline_terminal_retained):
    # The filesystem producer is real. The receipts/native counters are still
    # explicitly authored observations, without a model or provider invocation.
    from ipfs_accelerate_py.agent_supervisor.runtime.repository_world_attempt_journal import RepositoryWorldAttemptJournal

    declaration, census, _ = _population()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index, record in enumerate(census["slots"]):
        root = tmp_path / ("attempt-%d" % index)
        root.mkdir(mode=0o700)
        journal = RepositoryWorldAttemptJournal(root, workspace=workspace, start=record["start"])
        try:
            journal.retain_receipt(record["receipt"])
            if index != 0 or baseline_terminal_retained:
                journal.finish(outcome="returned")
            # These producer-retained descriptors, rather than hashes selected
            # from the subsequently read data, supply component identity pins.
            retained = journal.receipt()
        finally:
            journal.close()
        for component in ("start", "receipt", "terminal"):
            descriptor = retained[component]
            if descriptor is None:
                assert not (root / (component + ".json")).exists()
                record[component] = record[component + "_sha256"] = None
                continue
            raw = (root / descriptor["name"]).read_bytes()
            assert len(raw) == descriptor["bytes"]
            assert hashlib.sha256(raw).hexdigest() == descriptor["sha256"]
            record[component] = json.loads(raw)
            record[component + "_sha256"] = descriptor["sha256"]
        # Neither the journal nor a returned provider receipt observes a
        # generated file. A verifier must later bind an explicit output digest.
        record["output_sha256"] = None
    result = _compare(declaration, census)
    assert result["durable_population_gate"]["qualified"] is baseline_terminal_retained
    assert result["accounting_report"]["native_usage_by_arm"]["baseline"]["reported_native_token_subtotal"] == 120
    assert result["accounting_report"]["native_usage_by_arm"]["candidate"]["reported_native_token_subtotal"] == 80
    assert all(row["output_sha256"] is None for row in result["collection"]["attempts"])
    assert result["accounting_report"]["quality_gate"]["complete"] is False
    assert result["comparison_gate"]["observed_efficiency_qualified"] is False
    assert result["collection"]["collection_coverage"]["stdout_parsed"] is False
