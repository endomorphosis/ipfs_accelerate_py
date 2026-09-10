"""ASEH-010 closed telemetry receipts and paired-benchmark manifests."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    COMPUTE_FIELDS,
    COST_FIELDS,
    DIRECT_ARM_CONSTRAINTS,
    EQUAL_CONTROL_FIELDS,
    EVIDENCE_STATE_FIELDS,
    EfficiencyReceiptError,
    HISTORICAL_OUTCOMES,
    IDENTITY_FIELDS,
    MODEL_CLASSES,
    MODEL_USE_FIELDS,
    PAIRED_ARMS,
    PAIRED_BENCHMARK_MANIFEST_SCHEMA,
    PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH,
    QUALITY_FIELDS,
    SEALED_CURRENT_ARM_CONSTRAINTS,
    STATISTIC_FIELDS,
    TASK_EFFICIENCY_RECEIPT_SCHEMA,
    TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH,
    TERMINAL_FIELDS,
    TRUTH_STATES,
    WORK_FIELDS,
    admit_paired_benchmark_manifest,
    admit_task_efficiency_receipt,
    attempted_operation,
    build_paired_benchmark_manifest,
    build_task_efficiency_receipt,
    canonical_bytes,
    collect_unavailable_fields,
    content_identity,
    estimated_quantity,
    load_paired_benchmark_manifest_schema,
    load_task_efficiency_receipt_schema,
    measured_quantity,
    observed_operation,
    round_trip_paired_benchmark_manifest,
    round_trip_task_efficiency_receipt,
    simulated,
    unavailable,
    verified_operation,
)


COMMIT = "755f45475cc2d13dacd8b330036c1d597afeddde"
TREE = "729da9f8293ecfa046a0136381a3d3808f9ed140"
POLICY = "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
OBJECTIVE_REVISION = "baguqeeray6iu7h6kiajjow44w3l6gexkiihui2423wrhtpou22thcm6sqhbq"
VERIFIER_CID = content_identity({"verifier": "aseh-010"})
TASK_CID = content_identity({"task": "ASEH-010"})


def _cid(label: str) -> str:
    return content_identity({"aseh-010": label})


def _walk_object_schemas(node: Any, *, path: str = "$") -> list[str]:
    open_paths: list[str] = []
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            if node.get("additionalProperties") is not False:
                open_paths.append(path)
        for key, child in node.items():
            open_paths.extend(_walk_object_schemas(child, path=f"{path}.{key}"))
    elif isinstance(node, list):
        for index, child in enumerate(node):
            open_paths.extend(_walk_object_schemas(child, path=f"{path}[{index}]"))
    return open_paths


def _receipt(**overrides: Any):
    kwargs: dict[str, Any] = {
        "task_id": "ASEH-010",
        "task_cid": TASK_CID,
        "objective_id": "ASEH-G020",
        "objective_revision": OBJECTIVE_REVISION,
        "repository_commit": COMMIT,
        "repository_tree": TREE,
        "policy_identity": POLICY,
        "provider_id": "grok_cli",
        "model_id": "grok-4.6",
        "model_class": "remote_frontier_model",
        "final_task_outcome": "succeeded",
        "patch_disposition": "accepted",
        "population_kind": "hermetic_development",
        "live": False,
    }
    kwargs.update(overrides)
    return build_task_efficiency_receipt(**kwargs)


def _manifest(**overrides: Any):
    kwargs: dict[str, Any] = {
        "objective_revision": OBJECTIVE_REVISION,
        "repository_commit": COMMIT,
        "repository_tree": TREE,
        "policy_identity": POLICY,
        "task_inputs": _cid("task-inputs"),
        "acceptance_tests": _cid("acceptance-tests"),
        "available_providers_and_models": _cid("providers"),
        "price_accounting": _cid("price"),
        "resource_limits": _cid("limits"),
        "human_intervention_policy": _cid("human"),
    }
    kwargs.update(overrides)
    return build_paired_benchmark_manifest(**kwargs)


def test_schema_files_exist_and_are_closed() -> None:
    receipt_schema = load_task_efficiency_receipt_schema()
    manifest_schema = load_paired_benchmark_manifest_schema()
    assert TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH.is_file()
    assert PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH.is_file()
    assert receipt_schema["$id"] == TASK_EFFICIENCY_RECEIPT_SCHEMA
    assert manifest_schema["$id"] == PAIRED_BENCHMARK_MANIFEST_SCHEMA
    assert _walk_object_schemas(receipt_schema) == []
    assert _walk_object_schemas(manifest_schema) == []


def test_schema_documents_cover_required_field_groups() -> None:
    receipt = load_task_efficiency_receipt_schema()
    defs = receipt["$defs"]
    assert tuple(defs["identity"]["required"]) == IDENTITY_FIELDS
    assert tuple(defs["modelUse"]["required"]) == MODEL_USE_FIELDS
    assert tuple(defs["compute"]["required"]) == COMPUTE_FIELDS
    assert tuple(defs["cost"]["required"]) == COST_FIELDS
    assert tuple(defs["work"]["required"]) == WORK_FIELDS
    assert tuple(defs["quality"]["required"]) == QUALITY_FIELDS
    assert tuple(defs["evidenceState"]["required"]) == EVIDENCE_STATE_FIELDS
    assert tuple(defs["terminal"]["required"]) == TERMINAL_FIELDS
    assert tuple(defs["callsByModelClass"]["required"]) == MODEL_CLASSES
    manifest = load_paired_benchmark_manifest_schema()
    mdefs = manifest["$defs"]
    assert tuple(mdefs["arms"]["required"]) == PAIRED_ARMS
    assert tuple(mdefs["equalControls"]["required"]) == EQUAL_CONTROL_FIELDS
    assert tuple(mdefs["statistics"]["required"]) == STATISTIC_FIELDS


def test_receipt_round_trips_canonical_bytes() -> None:
    receipt = _receipt()
    encoded = receipt.canonical_bytes()
    assert encoded == canonical_bytes(json.loads(encoded.decode("utf-8")))
    replayed = admit_task_efficiency_receipt(encoded)
    assert replayed.canonical_bytes() == encoded
    assert replayed.receipt_cid == receipt.receipt_cid
    assert round_trip_task_efficiency_receipt(encoded) == encoded
    pretty = json.dumps(json.loads(encoded.decode("utf-8")), indent=2)
    assert admit_task_efficiency_receipt(pretty).canonical_bytes() == encoded


def test_manifest_round_trips_canonical_bytes() -> None:
    manifest = _manifest()
    encoded = manifest.canonical_bytes()
    replayed = admit_paired_benchmark_manifest(json.loads(encoded.decode("utf-8")))
    assert replayed.canonical_bytes() == encoded
    assert replayed.manifest_cid == content_identity(
        {key: value for key, value in replayed.to_dict().items() if key != "manifest_cid"}
    )
    assert round_trip_paired_benchmark_manifest(encoded) == encoded


def test_missing_values_remain_unavailable_and_never_zero() -> None:
    receipt = _receipt()
    unavailable_paths = receipt.explicit_unavailable_fields
    assert unavailable_paths
    assert "model_use.input_tokens" in unavailable_paths
    assert "compute.gpu_seconds" in unavailable_paths
    assert "cost.local_compute_units" in unavailable_paths
    payload = receipt.to_dict()
    for _path, node, state in _iter(payload):
        if state == "unavailable":
            assert "value" not in node
            assert "count" not in node
            assert node.get("reason_code")
            assert all(type(child) is not int or child != 0 for child in node.values())


def _iter(payload: Any):
    from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
        iter_evidence_nodes,
    )

    return iter_evidence_nodes(payload)


def test_receipt_distinguishes_all_required_truth_states() -> None:
    receipt = _receipt(
        model_use={
            "input_tokens": measured_quantity(12, unit="tokens", sensor_id="provider:grok"),
            "output_tokens": unavailable("provider_omitted"),
        },
        cost={
            "token_based_estimated_charge": estimated_quantity(
                3400,
                unit="microusd",
                estimator_id="aseh-price-snapshot",
                method="token_price_snapshot",
                price_snapshot_identity="unavailable",
            ),
            "provider_reported_charge": unavailable("provider_omitted"),
        },
        work={
            "tests_selected": observed_operation(
                "pytest-select", observer_id="validation-runner", count=4
            ),
            "tests_executed": verified_operation(
                "pytest",
                observer_id="validation-runner",
                count=4,
                verifier_id="pytest-junit",
                verifier_receipt_cid=VERIFIER_CID,
            ),
            "proof_obligations_executed": attempted_operation("lean-kernel", attempt_count=1),
            "validation_result": verified_operation(
                "validation",
                observer_id="validation-runner",
                count=1,
                verifier_id="pytest",
                verifier_receipt_cid=VERIFIER_CID,
            ),
        },
        compute={
            "concurrency": simulated("hermetic-lane-width", "fixture_only"),
            "gpu_seconds": unavailable("hardware_absent"),
        },
    )
    states = {state for _path, _node, state in _iter(receipt.to_dict())}
    for required in ("measured", "estimated", "unavailable", "attempted", "observed", "verified"):
        assert required in states
    assert "simulated" in states
    assert set(TRUTH_STATES) >= states
    assert receipt.model_use["input_tokens"]["truth_state"] == "measured"
    assert receipt.cost["token_based_estimated_charge"]["truth_state"] == "estimated"
    assert receipt.compute["gpu_seconds"]["truth_state"] == "unavailable"
    assert receipt.work["proof_obligations_executed"]["truth_state"] == "attempted"
    assert receipt.work["tests_selected"]["truth_state"] == "observed"
    assert receipt.work["tests_executed"]["truth_state"] == "verified"
    assert receipt.compute["concurrency"]["truth_state"] == "simulated"


def test_unknown_fields_fail_closed() -> None:
    payload = _receipt().to_dict()
    payload["unexpected_metric"] = 1
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        admit_task_efficiency_receipt(payload)
    nested = _receipt().to_dict()
    nested["compute"]["cpu_seconds"]["extra"] = "nope"
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(nested)
    manifest = _manifest().to_dict()
    manifest["arms"]["fourth_arm"] = {"arm_id": "other"}
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        admit_paired_benchmark_manifest(manifest)


def test_bounds_and_type_errors_fail_closed() -> None:
    with pytest.raises(EfficiencyReceiptError):
        measured_quantity(-1, unit="tokens", sensor_id="sensor")
    with pytest.raises(EfficiencyReceiptError):
        measured_quantity(True, unit="tokens", sensor_id="sensor")  # type: ignore[arg-type]
    payload = _receipt(
        model_use={
            "input_tokens": measured_quantity(1, unit="tokens", sensor_id="sensor"),
        }
    ).to_dict()
    payload["model_use"]["input_tokens"]["value"] = 10**18 + 1
    with pytest.raises(EfficiencyReceiptError, match="maximum bound|integer bound"):
        admit_task_efficiency_receipt(payload)
    payload = _receipt(
        model_use={
            "input_tokens": measured_quantity(1, unit="tokens", sensor_id="sensor"),
        }
    ).to_dict()
    payload["model_use"]["input_tokens"]["value"] = 1.5
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)
    payload = _receipt().to_dict()
    payload["identity"]["repository_commit"] = "not-a-git-oid"
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)
    with pytest.raises(EfficiencyReceiptError):
        build_paired_benchmark_manifest(
            objective_revision=OBJECTIVE_REVISION,
            repository_commit=COMMIT,
            repository_tree=TREE,
            policy_identity=POLICY,
            task_inputs=_cid("task-inputs"),
            acceptance_tests=_cid("acceptance-tests"),
            available_providers_and_models=_cid("providers"),
            price_accounting=_cid("price"),
            resource_limits=_cid("limits"),
            human_intervention_policy=_cid("human"),
            maximum_retries=99,
        )


def test_estimated_cannot_be_admitted_as_measured() -> None:
    payload = _receipt(
        cost={
            "token_based_estimated_charge": estimated_quantity(
                12,
                unit="microusd",
                estimator_id="price",
                method="token_price_snapshot",
            )
        }
    ).to_dict()
    payload["cost"]["token_based_estimated_charge"]["truth_state"] = "measured"
    payload["cost"]["token_based_estimated_charge"]["sensor_id"] = "sensor"
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)
    measured_as_estimate = _receipt(
        cost={
            "provider_reported_charge": measured_quantity(
                9, unit="microusd", sensor_id="provider"
            )
        }
    ).to_dict()
    measured_as_estimate["cost"]["provider_reported_charge"]["truth_state"] = "estimated"
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(measured_as_estimate)


def test_unavailable_cannot_encode_numeric_zero() -> None:
    payload = _receipt().to_dict()
    payload["compute"]["gpu_seconds"] = {
        "truth_state": "unavailable",
        "reason_code": "hardware_absent",
        "value": 0,
    }
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)
    payload = _receipt().to_dict()
    payload["cost"]["local_compute_units"] = {
        "truth_state": "unavailable",
        "reason_code": "not_reported",
        "count": 0,
    }
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)


def test_attempted_cannot_be_represented_as_observed() -> None:
    payload = _receipt(
        work={"proof_obligations_executed": attempted_operation("lean", attempt_count=2)}
    ).to_dict()
    payload["work"]["proof_obligations_executed"]["truth_state"] = "observed"
    payload["work"]["proof_obligations_executed"]["observer_id"] = "runner"
    payload["work"]["proof_obligations_executed"]["count"] = 2
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)


def test_observed_cannot_be_represented_as_verified_without_verifier() -> None:
    payload = _receipt(
        work={
            "tests_executed": observed_operation(
                "pytest", observer_id="runner", count=3
            )
        }
    ).to_dict()
    payload["work"]["tests_executed"]["truth_state"] = "verified"
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)
    payload = _receipt(
        work={
            "tests_executed": observed_operation(
                "pytest", observer_id="runner", count=3
            )
        }
    ).to_dict()
    payload["work"]["tests_executed"]["verifier_id"] = "pytest"
    payload["work"]["tests_executed"]["verifier_receipt_cid"] = VERIFIER_CID
    with pytest.raises(EfficiencyReceiptError):
        admit_task_efficiency_receipt(payload)


def test_simulated_cannot_be_represented_as_live() -> None:
    with pytest.raises(EfficiencyReceiptError, match="live"):
        _receipt(
            population_kind="new_live_shadow_canary",
            live=True,
            compute={"concurrency": simulated("fixture", "fixture_only")},
        )
    with pytest.raises(EfficiencyReceiptError, match="hermetic"):
        _receipt(population_kind="hermetic_development", live=True)


def test_measured_zero_requires_a_sensor_and_is_not_unavailable() -> None:
    receipt = _receipt(
        model_use={
            "cached_input_tokens": measured_quantity(
                0, unit="tokens", sensor_id="provider:grok"
            )
        }
    )
    sample = receipt.model_use["cached_input_tokens"]
    assert sample["truth_state"] == "measured"
    assert sample["value"] == 0
    assert sample["sensor_id"]
    assert "model_use.cached_input_tokens" not in receipt.explicit_unavailable_fields


def test_receipt_cid_mismatch_and_missing_identity_fail_closed() -> None:
    payload = _receipt().to_dict()
    payload["receipt_cid"] = VERIFIER_CID
    with pytest.raises(EfficiencyReceiptError, match="content identity"):
        admit_task_efficiency_receipt(payload)
    payload = _receipt().to_dict()
    del payload["identity"]["task_id"]
    with pytest.raises(EfficiencyReceiptError, match="missing required"):
        admit_task_efficiency_receipt(payload)


def test_explicit_unavailable_index_must_match_payload() -> None:
    payload = _receipt().to_dict()
    payload["explicit_unavailable_fields"] = []
    with pytest.raises(EfficiencyReceiptError, match="explicit_unavailable_fields"):
        admit_task_efficiency_receipt(payload)
    payload = _receipt().to_dict()
    payload["explicit_unavailable_fields"] = list(payload["explicit_unavailable_fields"]) + [
        "model_use.made_up"
    ]
    with pytest.raises(EfficiencyReceiptError, match="explicit_unavailable_fields"):
        admit_task_efficiency_receipt(payload)


def test_paired_manifest_requires_three_arms_and_equal_controls() -> None:
    manifest = _manifest()
    payload = manifest.to_dict()
    assert set(payload["arms"]) == set(PAIRED_ARMS)
    assert payload["arms"]["candidate_optimized_supervisor"]["shadow_before_mutation"] is True
    assert payload["arms"]["direct_minimal_orchestration_baseline"]["constraints"] == list(
        DIRECT_ARM_CONSTRAINTS
    )
    assert payload["arms"]["sealed_current_supervisor_baseline"]["constraints"] == list(
        SEALED_CURRENT_ARM_CONSTRAINTS
    )
    assert payload["equal_controls"]["repository_revision"] == TREE
    assert payload["populations"]["hermetic_development"]["minimum"] == 60
    assert payload["populations"]["historical_exact_tree_replay"]["minimum"] == 20
    assert payload["populations"]["new_live_shadow_canary"]["minimum"] == 10
    assert payload["populations"]["historical_exact_tree_replay"]["required_outcomes"] == list(
        HISTORICAL_OUTCOMES
    )
    assert payload["promotion_without_paired_campaign"] is False
    assert payload["hermetic_sufficient_for_production_promotion"] is False
    for field in STATISTIC_FIELDS:
        assert payload["statistics"][field]["truth_state"] == "unavailable"
    assert payload["equal_controls"]["maximum_retries"] == 3


def test_paired_manifest_rejects_open_statistics_and_promotion_bypass() -> None:
    payload = _manifest().to_dict()
    payload["promotion_without_paired_campaign"] = True
    with pytest.raises(EfficiencyReceiptError):
        admit_paired_benchmark_manifest(payload)
    payload = _manifest().to_dict()
    payload["hermetic_sufficient_for_production_promotion"] = True
    with pytest.raises(EfficiencyReceiptError):
        admit_paired_benchmark_manifest(payload)
    payload = _manifest().to_dict()
    payload["statistics"]["median_difference"] = {
        "truth_state": "unavailable",
        "reason_code": "not_yet_measured",
        "value": 0,
    }
    with pytest.raises(EfficiencyReceiptError):
        admit_paired_benchmark_manifest(payload)


def test_schema_truth_state_variants_are_closed_and_distinct() -> None:
    defs = load_task_efficiency_receipt_schema()["$defs"]
    quantity_states = {
        variant["$ref"].rsplit("/", 1)[-1]
        for variant in defs["quantityObservation"]["oneOf"]
    }
    operation_states = {
        variant["$ref"].rsplit("/", 1)[-1]
        for variant in defs["operationObservation"]["oneOf"]
    }
    assert "measuredQuantity" in quantity_states
    assert "estimatedQuantity" in quantity_states
    assert "unavailableEvidence" in quantity_states
    assert "attemptedOperation" in operation_states
    assert "observedOperation" in operation_states
    assert "verifiedOperation" in operation_states
    assert defs["measuredQuantity"]["properties"]["truth_state"]["const"] == "measured"
    assert defs["estimatedQuantity"]["properties"]["truth_state"]["const"] == "estimated"
    assert defs["unavailableEvidence"]["properties"]["truth_state"]["const"] == "unavailable"
    assert defs["attemptedOperation"]["properties"]["truth_state"]["const"] == "attempted"
    assert defs["observedOperation"]["properties"]["truth_state"]["const"] == "observed"
    assert defs["verifiedOperation"]["properties"]["truth_state"]["const"] == "verified"
    assert "value" not in defs["unavailableEvidence"]["properties"]
    assert "verifier_receipt_cid" in defs["verifiedOperation"]["required"]
    assert "verifier_receipt_cid" not in defs["observedOperation"]["properties"]


def test_work_quality_and_terminal_fields_are_admitted() -> None:
    receipt = _receipt()
    for name in WORK_FIELDS:
        assert name in receipt.work
    for name in QUALITY_FIELDS:
        assert name in receipt.quality
    for name in TERMINAL_FIELDS:
        assert name in receipt.terminal
    assert receipt.terminal["terminalized"] is True
    assert receipt.evidence_state["unavailable_as_zero"] is False
    assert receipt.identity["interface_schema_identities"]


def test_collect_unavailable_fields_is_stable() -> None:
    receipt = _receipt()
    again = collect_unavailable_fields(
        {key: value for key, value in receipt.to_dict().items() if key != "receipt_cid"}
    )
    assert again == receipt.explicit_unavailable_fields
    assert all(item == item.lower() for item in again)
    assert again == tuple(sorted(again))
