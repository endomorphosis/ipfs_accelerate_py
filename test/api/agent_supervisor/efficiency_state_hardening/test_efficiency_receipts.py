"""Contract tests for canonical efficiency telemetry records."""

from __future__ import annotations

import json

import jsonschema
import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    EfficiencyReceiptError,
    PAIRED_BENCHMARK_MANIFEST_INTERFACE,
    PAIRED_BENCHMARK_MANIFEST_SCHEMA,
    TASK_EFFICIENCY_RECEIPT_INTERFACE,
    TASK_EFFICIENCY_RECEIPT_SCHEMA,
    PairedBenchmarkManifest,
    TaskEfficiencyReceipt,
    TruthState,
    schema_path,
)


def _evidence(state: str, value: object = 1, source: str = "instrumented") -> dict[str, object]:
    if state == TruthState.UNAVAILABLE.value:
        return {"state": state, "value": None, "source": "unavailable", "reference_ids": []}
    references = ["evidence:verified-1"] if state == TruthState.VERIFIED.value else []
    return {"state": state, "value": value, "source": source, "reference_ids": references}


def _receipt_payload() -> dict[str, object]:
    unavailable = ["cached_input_tokens_where_reported", "gpu_seconds"]
    model = {
        "input_tokens": _evidence("measured", 120),
        "output_tokens": _evidence("measured", 34),
        "cached_input_tokens_where_reported": _evidence("unavailable"),
        "reasoning_tokens_where_reported": _evidence("estimated", 8, "token-estimator"),
        "number_of_calls": _evidence("measured", 2),
        "calls_by_model_class": _evidence("measured", [{"model_class": "remote_standard_model", "count": 2}]),
        "safe_provider_request_ids": _evidence("observed", ["request:safe-1"]),
        "provider_reported_usage_where_available": _evidence("measured", {"input_tokens": 120, "output_tokens": 34}, "provider"),
        "explicit_unavailable_fields": _evidence("observed", unavailable),
    }
    compute_values = {
        "cpu_seconds": 12, "gpu_seconds": None, "peak_memory_bytes": 4096,
        "wall_clock_duration_ms": 18000, "test_execution_time_ms": 12000,
        "prover_execution_time_ms": 0, "static_analysis_time_ms": 1000,
        "indexing_retrieval_time_ms": 500, "bytes_read": 2048, "bytes_written": 1024,
        "process_count": 3, "concurrency": 2,
    }
    compute = {name: _evidence("unavailable") if value is None else _evidence("measured", value) for name, value in compute_values.items()}
    cost = {
        "provider_reported_charge_microunits": _evidence("measured", 90, "provider"),
        "token_based_estimated_charge_microunits": _evidence("estimated", 95, "price-snapshot"),
        "price_snapshot_identity_and_timestamp": _evidence("observed", "prices:2026-08-24"),
        "local_compute_units": _evidence("measured", 12),
        "audit_and_verification_overhead_microunits": _evidence("measured", 3),
        "cost_per_accepted_patch_microunits": _evidence("estimated", 93, "calculator"),
        "cost_per_completed_task_microunits": _evidence("estimated", 93, "calculator"),
    }
    work = {
        "tests_selected": _evidence("observed", ["test_a"]), "tests_executed": _evidence("measured", ["test_a"]),
        "full_suite_tests": _evidence("attempted", ["pytest -q"]), "type_static_schema_checks": _evidence("verified", ["schema-check"]),
        "proof_obligations_selected": _evidence("observed", ["proof:one"]), "proof_obligations_executed": _evidence("attempted", ["proof:one"]),
        "proof_receipts_reused": _evidence("measured", ["receipt:prior"]), "retries": _evidence("measured", 0),
        "rescue_attempts": _evidence("measured", 0), "merge_conflicts": _evidence("measured", 0),
        "manual_recovery": _evidence("observed", []), "human_interventions": _evidence("observed", []),
        "final_task_outcome": _evidence("observed", "completed"), "validation_result": _evidence("verified", "passed"),
        "patch_disposition": _evidence("observed", "accepted"),
    }
    return {
        "schema": TASK_EFFICIENCY_RECEIPT_SCHEMA,
        "interface": TASK_EFFICIENCY_RECEIPT_INTERFACE,
        "identity": {
            "task_id": "ASEH-010", "task_cid": "bafy-task", "objective_id": "ASEH-G020", "objective_revision": "objective:1",
            "repository_commit": "abc123", "repository_tree": "tree123", "policy_identity": "policy:1", "context_pack_cid": "bafy-context",
            "interface_schema_identities": [TASK_EFFICIENCY_RECEIPT_SCHEMA],
            "provider_model_exact_identity_and_revision": [{"provider": "provider", "model": "model", "revision": "2026-08-24"}],
        },
        "model_use": model, "compute": compute, "cost": cost, "work": work,
        "quality": {"acceptance_tests_passed": _evidence("verified", 1), "quality_gate_result": _evidence("verified", "passed"), "accepted_patch_rate_basis": _evidence("observed", {"accepted": 1, "considered": 1})},
        "evidence": {"attempted_operations": _evidence("attempted", ["pytest"]), "observed_effects": _evidence("observed", ["exit:0"]), "verified_evidence": _evidence("verified", ["receipt:validation"])},
        "terminal": {"is_terminal": True, "outcome": "completed", "evidence_state": "verified", "terminal_reason": "validation passed"},
    }


def _manifest_payload() -> dict[str, object]:
    return {
        "schema": PAIRED_BENCHMARK_MANIFEST_SCHEMA, "interface": PAIRED_BENCHMARK_MANIFEST_INTERFACE, "manifest_id": "campaign:1",
        "baseline": {"repository_revision": "commit:1", "repository_tree": "tree:1", "objective_id": "ASEH-G020", "objective_revision": "objective:1", "task_inputs_cid": "bafy-inputs", "acceptance_tests_cid": "bafy-tests", "available_provider_models": ["provider/model@revision"], "price_accounting": "prices:1", "resource_limits": _evidence("observed", {"wall_ms": 1000}), "maximum_retries": 2, "human_intervention_policy": "policy:human:1"},
        "arms": [
            {"arm": "candidate_optimized_supervisor", "constraints": ["campaign_changes_only", "shadow_before_mutation"], "receipt_ids": ["receipt:candidate"], "evidence_state": "measured"},
            {"arm": "direct_minimal_orchestration_baseline", "constraints": ["same_safe_isolation_and_acceptance_tests", "no_candidate_context_pack_optimization", "no_candidate_proof_or_test_reuse_optimization", "same_available_model_and_provider_set", "no_unsafe_direct_mutation"], "receipt_ids": ["receipt:direct"], "evidence_state": "measured"},
            {"arm": "sealed_current_supervisor_baseline", "constraints": ["exact_pre_campaign_policy", "exact_pre_campaign_routing_behavior"], "receipt_ids": ["receipt:sealed"], "evidence_state": "measured"},
        ],
        "equal_controls": ["repository_revision", "objective", "task_inputs", "acceptance_tests", "available_providers_and_models", "price_accounting", "resource_limits", "maximum_retries", "human_intervention_policy"],
        "populations": {"hermetic_development_count": 60, "historical_exact_tree_replay_count": 20, "new_live_shadow_canary_count": 10, "live_enrollment_deadline_days": 30},
        "historical_required_outcomes": ["successful", "failed", "retried", "rescued", "conflicted", "human_escalated"],
        "statistics": ["median_difference", "bootstrap_confidence_intervals"], "promotion_without_paired_campaign": False,
    }


def test_task_receipt_round_trips_exact_canonical_bytes() -> None:
    receipt = TaskEfficiencyReceipt(_receipt_payload())
    encoded = receipt.canonical_bytes()
    assert encoded == receipt.canonical_json().encode("utf-8")
    assert TaskEfficiencyReceipt.from_canonical_bytes(encoded).canonical_bytes() == encoded
    with pytest.raises(EfficiencyReceiptError, match="not canonical"):
        TaskEfficiencyReceipt.from_canonical_bytes(b" " + encoded)
    with pytest.raises(EfficiencyReceiptError, match="duplicate"):
        TaskEfficiencyReceipt.from_canonical_bytes(
            b'{"schema":"x","schema":"x"}'
        )
    assert receipt.to_dict()["model_use"]["cached_input_tokens_where_reported"]["state"] == "unavailable"  # type: ignore[index]


def test_receipt_rejects_unknown_fields_noncanonical_numbers_and_truth_state_confusion() -> None:
    unknown = _receipt_payload()
    unknown["unexpected"] = True
    with pytest.raises(EfficiencyReceiptError, match="unknown"):
        TaskEfficiencyReceipt(unknown)
    noncanonical = _receipt_payload()
    noncanonical["compute"]["cpu_seconds"]["value"] = 1.5  # type: ignore[index]
    with pytest.raises(EfficiencyReceiptError, match="float"):
        TaskEfficiencyReceipt(noncanonical)
    confused = _receipt_payload()
    confused["model_use"]["cached_input_tokens_where_reported"] = {  # type: ignore[index]
        "state": "unavailable", "value": 0, "source": "unavailable", "reference_ids": []
    }
    with pytest.raises(EfficiencyReceiptError, match="unavailable evidence"):
        TaskEfficiencyReceipt(confused)
    missing_unavailable = _receipt_payload()
    missing_unavailable["model_use"]["explicit_unavailable_fields"]["value"] = []  # type: ignore[index]
    with pytest.raises(EfficiencyReceiptError, match="exactly name"):
        TaskEfficiencyReceipt(missing_unavailable)
    inflated = _receipt_payload()
    inflated["evidence"]["attempted_operations"]["state"] = "observed"  # type: ignore[index]
    with pytest.raises(EfficiencyReceiptError, match="must be attempted"):
        TaskEfficiencyReceipt(inflated)


def test_verified_evidence_requires_admitted_reference_and_terminal_cannot_be_unavailable() -> None:
    payload = _receipt_payload()
    payload["quality"]["acceptance_tests_passed"]["reference_ids"] = []  # type: ignore[index]
    with pytest.raises(EfficiencyReceiptError, match="verified evidence requires"):
        TaskEfficiencyReceipt(payload)
    payload = _receipt_payload()
    payload["terminal"]["evidence_state"] = "unavailable"  # type: ignore[index]
    with pytest.raises(EfficiencyReceiptError, match="cannot be unavailable"):
        TaskEfficiencyReceipt(payload)


def test_paired_manifest_canonicalizes_arm_order_and_rejects_bounds_or_unknowns() -> None:
    manifest = PairedBenchmarkManifest(_manifest_payload())
    decoded = json.loads(manifest.canonical_bytes())
    assert [item["arm"] for item in decoded["arms"]] == [
        "direct_minimal_orchestration_baseline", "sealed_current_supervisor_baseline", "candidate_optimized_supervisor"
    ]
    assert PairedBenchmarkManifest.from_canonical_bytes(manifest.canonical_bytes()) == manifest
    bad_deadline = _manifest_payload()
    bad_deadline["populations"]["live_enrollment_deadline_days"] = 31  # type: ignore[index]
    with pytest.raises(EfficiencyReceiptError, match="30 days"):
        PairedBenchmarkManifest(bad_deadline)
    bad_arm = _manifest_payload()
    bad_arm["arms"][0]["unknown"] = True  # type: ignore[index]
    with pytest.raises(EfficiencyReceiptError, match="unknown"):
        PairedBenchmarkManifest(bad_arm)


def test_declared_schemas_are_closed_machine_readable_contracts() -> None:
    receipt_schema = json.loads(schema_path("task_efficiency_receipt.schema.json").read_text())
    manifest_schema = json.loads(schema_path("paired_benchmark_manifest.schema.json").read_text())
    assert receipt_schema["additionalProperties"] is False
    assert manifest_schema["additionalProperties"] is False
    jsonschema.Draft202012Validator.check_schema(receipt_schema)
    jsonschema.Draft202012Validator.check_schema(manifest_schema)
    receipt_validator = jsonschema.Draft202012Validator(receipt_schema)
    receipt_validator.validate(_receipt_payload())
    bad_unknown = _receipt_payload()
    bad_unknown["unknown"] = True
    with pytest.raises(jsonschema.ValidationError):
        receipt_validator.validate(bad_unknown)
    bad_unavailable = _receipt_payload()
    bad_unavailable["compute"]["gpu_seconds"] = {  # type: ignore[index]
        "state": "unavailable", "value": 0, "source": "unavailable", "reference_ids": []
    }
    with pytest.raises(jsonschema.ValidationError):
        receipt_validator.validate(bad_unavailable)
    bad_evidence_state = _receipt_payload()
    bad_evidence_state["evidence"]["observed_effects"]["state"] = "verified"  # type: ignore[index]
    with pytest.raises(jsonschema.ValidationError):
        receipt_validator.validate(bad_evidence_state)
    manifest_validator = jsonschema.Draft202012Validator(manifest_schema)
    manifest_validator.validate(_manifest_payload())
    bad_bound = _manifest_payload()
    bad_bound["populations"]["live_enrollment_deadline_days"] = 31  # type: ignore[index]
    with pytest.raises(jsonschema.ValidationError):
        manifest_validator.validate(bad_bound)
    assert set(TruthState) == {
        TruthState.MEASURED, TruthState.ESTIMATED, TruthState.UNAVAILABLE,
        TruthState.ATTEMPTED, TruthState.OBSERVED, TruthState.VERIFIED, TruthState.SIMULATED,
    }
