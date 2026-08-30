"""ASEH-010 closed telemetry receipt and paired-benchmark manifest contracts."""

from __future__ import annotations

import copy
import json
from dataclasses import replace
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    canonical_json_bytes,
)
from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    DISTINGUISHED_TRUTH_STATES,
    EfficiencyReceiptError,
    EqualControls,
    FinalTaskOutcome,
    IdentityEvidence,
    PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH,
    PatchDisposition,
    PairedBenchmarkManifest,
    PopulationKind,
    Populations,
    QuantitativeEvidence,
    ReceiptIdentity,
    RequestIdEvidence,
    TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH,
    TRUTH_STATES,
    TaskEfficiencyReceipt,
    TerminalRecord,
    UNIT_COUNT,
    UNIT_MICROUSD,
    UNIT_TOKENS,
    ValidationResult,
    WorkUse,
    admit_paired_benchmark_manifest,
    admit_task_efficiency_receipt,
    fixture_cid,
    load_paired_benchmark_manifest_schema,
    load_task_efficiency_receipt_schema,
    sensor_id_for,
    unavailable_identity,
    unavailable_quantitative,
    unavailable_task_efficiency_receipt,
    validate_closed_schema,
)


COMMIT = "755f45475cc2d13dacd8b330036c1d597afeddde"
TREE = "729da9f8293ecfa046a0136381a3d3808f9ed140"


def _identity() -> ReceiptIdentity:
    return ReceiptIdentity(
        task_id="ASEH-010",
        task_cid=fixture_cid("task"),
        objective_id="ASEH-G020",
        objective_revision="baguqeeray6iu7h6kiajjow44w3l6gexkiihui2423wrhtpou22thcm6sqhbq",
        repository_commit=COMMIT,
        repository_tree=TREE,
        policy_identity="policy:implementation-daemon",
        context_pack_cid=unavailable_identity("identity.context_pack_cid"),
        interface_schema_identities=(
            "aseh/task-efficiency-receipt@1",
            "aseh/paired-benchmark-manifest@1",
        ),
        provider_id="grok_cli",
        model_id="grok-4.6",
        model_revision=unavailable_identity("identity.model_revision"),
    )


def _receipt(**overrides: Any) -> TaskEfficiencyReceipt:
    return unavailable_task_efficiency_receipt(identity=_identity(), **overrides)


def _equal_controls() -> EqualControls:
    return EqualControls(
        repository_revision=IdentityEvidence.observed(
            f"{COMMIT}:{TREE}",
            sensor_id=sensor_id_for("equal_controls.repository_revision"),
        ),
        objective=IdentityEvidence.observed(
            "ASEH-G020",
            sensor_id=sensor_id_for("equal_controls.objective"),
        ),
        task_inputs=unavailable_identity("equal_controls.task_inputs"),
        acceptance_tests=unavailable_identity("equal_controls.acceptance_tests"),
        available_providers_and_models=unavailable_identity(
            "equal_controls.available_providers_and_models"
        ),
        price_accounting=unavailable_identity("equal_controls.price_accounting"),
        resource_limits=unavailable_identity("equal_controls.resource_limits"),
        maximum_retries=unavailable_quantitative("equal_controls.maximum_retries"),
        human_intervention_policy=unavailable_identity(
            "equal_controls.human_intervention_policy"
        ),
    )


def _manifest(**overrides: Any) -> PairedBenchmarkManifest:
    payload = {
        "objective_id": "ASEH-G020",
        "objective_revision": "ASEH-PLAN-R1",
        "repository_commit": COMMIT,
        "repository_tree": TREE,
        "policy_identity": "policy:implementation-daemon",
        "sealed_input_manifest_identity": unavailable_identity(
            "sealed_input_manifest_identity", reason="not-sealed"
        ),
        "price_snapshot_identity": unavailable_identity("price_snapshot_identity"),
        "environment_identity": unavailable_identity("environment_identity"),
        "provider_model_config_identity": unavailable_identity(
            "provider_model_config_identity"
        ),
        "equal_controls": _equal_controls(),
        "populations": Populations(
            hermetic_sealed_count=unavailable_quantitative(
                "populations.hermetic_development.sealed_count", reason="not-sealed"
            ),
            replay_sealed_count=unavailable_quantitative(
                "populations.historical_exact_tree_replay.sealed_count",
                reason="not-sealed",
            ),
            live_enrolled_count=unavailable_quantitative(
                "populations.new_live_shadow_canary.enrolled_count",
                reason="not-sealed",
            ),
        ),
    }
    payload.update(overrides)
    return PairedBenchmarkManifest(**payload)


def _assert_closed_schema(node: Any) -> None:
    if isinstance(node, list):
        for item in node:
            _assert_closed_schema(item)
        return
    if not isinstance(node, dict):
        return
    if node.get("type") == "object" or "properties" in node or "required" in node:
        assert node.get("additionalProperties") is False
    for key in ("properties", "$defs"):
        child = node.get(key)
        if isinstance(child, dict):
            for item in child.values():
                _assert_closed_schema(item)
    for key in ("oneOf", "prefixItems"):
        child = node.get(key)
        if isinstance(child, list):
            _assert_closed_schema(child)
    items = node.get("items")
    if isinstance(items, dict):
        _assert_closed_schema(items)


def _distinguished_receipt() -> TaskEfficiencyReceipt:
    verifier = fixture_cid("verifier")
    snapshot = "baguqeera3nas3dp546nrxqg3nr6qdalxzbld2wlbhxzrdt5bz5disttd3w6a"
    base = _receipt(
        terminal=TerminalRecord(
            final_task_outcome=FinalTaskOutcome.FAILED,
            validation_result=ValidationResult.FAILED,
            patch_disposition=PatchDisposition.REJECTED,
            population_kind=PopulationKind.HERMETIC,
        )
    )
    model = replace(
        base.model,
        input_tokens=QuantitativeEvidence.measured(
            12,
            unit=UNIT_TOKENS,
            sensor_id=sensor_id_for("model.input_tokens", "measured"),
        ),
        output_tokens=unavailable_quantitative("model.output_tokens"),
    )
    compute = replace(
        base.compute,
        cpu_seconds=QuantitativeEvidence.observed(
            1_500_000,
            unit="seconds_millionths",
            sensor_id=sensor_id_for("compute.cpu_seconds", "observed"),
        ),
    )
    cost = replace(
        base.cost,
        token_based_estimated_charge=QuantitativeEvidence.estimated(
            2500,
            unit=UNIT_MICROUSD,
            estimate_method="token-price-snapshot",
            price_snapshot_identity=snapshot,
        ),
        price_snapshot_identity=IdentityEvidence.observed(
            snapshot,
            sensor_id=sensor_id_for("cost.price_snapshot_identity"),
        ),
    )
    work = replace(
        base.work,
        retries=QuantitativeEvidence.attempted(
            "retry",
            attempt_id="attempt:2",
        ),
        tests_executed=QuantitativeEvidence.verified(
            4,
            unit=UNIT_COUNT,
            verifier_identity="pytest",
            verifier_receipt_cid=verifier,
        ),
    )
    quality = replace(
        base.quality,
        false_positives=QuantitativeEvidence.simulated(
            0,
            unit=UNIT_COUNT,
            fixture_identity="hermetic-false-positives",
        ),
    )
    return TaskEfficiencyReceipt(
        identity=base.identity,
        model=model,
        compute=compute,
        cost=cost,
        work=work,
        quality=quality,
        terminal=base.terminal,
    )


def test_schema_files_exist_and_are_closed() -> None:
    receipt_schema = load_task_efficiency_receipt_schema()
    manifest_schema = load_paired_benchmark_manifest_schema()
    assert TASK_EFFICIENCY_RECEIPT_SCHEMA_PATH.is_file()
    assert PAIRED_BENCHMARK_MANIFEST_SCHEMA_PATH.is_file()
    assert receipt_schema["properties"]["schema"]["const"] == "aseh/task-efficiency-receipt@1"
    assert (
        manifest_schema["properties"]["schema"]["const"]
        == "aseh/paired-benchmark-manifest@1"
    )
    _assert_closed_schema(receipt_schema)
    _assert_closed_schema(manifest_schema)
    for state in DISTINGUISHED_TRUTH_STATES:
        assert state in TRUTH_STATES


def test_task_efficiency_receipt_round_trips_canonical_bytes() -> None:
    receipt = _distinguished_receipt()
    payload = receipt.canonical_bytes()
    admitted = admit_task_efficiency_receipt(payload)
    assert admitted.canonical_bytes() == payload
    assert TaskEfficiencyReceipt.from_dict(receipt.to_dict()).canonical_bytes() == payload
    assert admitted.content_id == receipt.content_id
    validate_closed_schema(receipt.to_dict(), load_task_efficiency_receipt_schema())
    assert receipt.to_dict()["authority"] is False
    assert receipt.to_dict()["attests_measurements"] is False
    assert receipt.to_dict()["attests_task_completion"] is False


def test_paired_benchmark_manifest_round_trips_canonical_bytes() -> None:
    manifest = _manifest()
    payload = manifest.canonical_bytes()
    admitted = admit_paired_benchmark_manifest(payload)
    assert admitted.canonical_bytes() == payload
    assert PairedBenchmarkManifest.from_dict(manifest.to_dict()).canonical_bytes() == payload
    validate_closed_schema(manifest.to_dict(), load_paired_benchmark_manifest_schema())
    assert admitted.to_dict()["promotion_without_paired_campaign"] is False
    assert admitted.to_dict()["pairing_rejects_unequal_controls"] is True
    assert [arm["arm_id"] for arm in admitted.to_dict()["arms"]] == [
        "direct_minimal_orchestration_baseline",
        "sealed_current_supervisor_baseline",
        "candidate_optimized_supervisor",
    ]


def test_missing_metrics_remain_unavailable_not_zero() -> None:
    receipt = _receipt()
    payload = receipt.to_dict()
    assert payload["model"]["input_tokens"] == {
        "status": "unavailable",
        "reason_code": "not-reported",
        "sensor_id": sensor_id_for("model.input_tokens", "not-reported"),
    }
    assert "value" not in payload["model"]["input_tokens"]
    assert "unit" not in payload["model"]["input_tokens"]
    assert payload["cost"]["local_compute_units"]["status"] == "unavailable"
    assert payload["evidence_state"]["unavailable"]
    assert payload["evidence_state"]["measured"] == []
    for field in payload["evidence_state"]["unavailable"]:
        assert field not in payload["evidence_state"]["measured"]


def test_distinguishes_measured_estimated_unavailable_attempted_observed_and_verified() -> None:
    receipt = _distinguished_receipt()
    state = receipt.evidence_state()
    assert state["measured"] == ["model.input_tokens"]
    assert state["estimated"] == ["cost.token_based_estimated_charge"]
    assert state["attempted"] == ["work.retries"]
    assert state["observed"] == [
        "compute.cpu_seconds",
        "cost.price_snapshot_identity",
    ]
    assert state["verified"] == ["work.tests_executed"]
    assert state["simulated"] == ["quality.false_positives"]
    assert "model.output_tokens" in state["unavailable"]
    bodies = {
        "measured": receipt.model.input_tokens.to_dict(),
        "estimated": receipt.cost.token_based_estimated_charge.to_dict(),
        "unavailable": receipt.model.output_tokens.to_dict(),
        "attempted": receipt.work.retries.to_dict(),
        "observed": receipt.compute.cpu_seconds.to_dict(),
        "verified": receipt.work.tests_executed.to_dict(),
    }
    assert {item["status"] for item in bodies.values()} == set(DISTINGUISHED_TRUTH_STATES)
    serialized = {name: canonical_json_bytes(body) for name, body in bodies.items()}
    assert len(set(serialized.values())) == len(DISTINGUISHED_TRUTH_STATES)
    assert "value" not in bodies["unavailable"]
    assert "value" not in bodies["attempted"]
    assert bodies["estimated"]["estimate_method"] == "token-price-snapshot"
    assert "verifier_receipt_cid" in bodies["verified"]
    assert "verifier_receipt_cid" not in bodies["observed"]


def test_rejects_unknown_fields() -> None:
    receipt = _receipt()
    payload = receipt.to_dict()
    payload["unexpected"] = "field"
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        TaskEfficiencyReceipt.from_dict(payload)
    nested = receipt.to_dict()
    nested["model"]["invented_tokens"] = nested["model"]["input_tokens"]
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        TaskEfficiencyReceipt.from_dict(nested)
    measured = receipt.model.input_tokens.to_dict()
    measured["status"] = "measured"
    measured["value"] = 1
    measured["unit"] = UNIT_TOKENS
    measured["sensor_id"] = sensor_id_for("model.input_tokens", "measured")
    measured["reason_code"] = "not-reported"
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        QuantitativeEvidence.from_dict(measured)
    manifest = _manifest().to_dict()
    manifest["extra"] = True
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        PairedBenchmarkManifest.from_dict(manifest)


def test_rejects_bounds_errors() -> None:
    with pytest.raises(EfficiencyReceiptError, match="between"):
        QuantitativeEvidence.measured(
            10**18 + 1,
            unit=UNIT_TOKENS,
            sensor_id=sensor_id_for("bound"),
        )
    with pytest.raises(EfficiencyReceiptError, match="between"):
        QuantitativeEvidence.measured(
            -1,
            unit=UNIT_TOKENS,
            sensor_id=sensor_id_for("bound"),
        )
    identity = _identity()
    with pytest.raises(EfficiencyReceiptError):
        replace(identity, task_id="")
    with pytest.raises(EfficiencyReceiptError):
        replace(identity, repository_commit="not-a-git-object")
    with pytest.raises(EfficiencyReceiptError):
        replace(identity, task_cid="cid-not-canonical")
    too_many = tuple(f"schema:{index}" for index in range(65))
    with pytest.raises(EfficiencyReceiptError, match="64-item"):
        replace(identity, interface_schema_identities=too_many)


def test_rejects_noncanonical_numeric_values() -> None:
    receipt = _distinguished_receipt()
    raw = json.loads(receipt.canonical_bytes().decode("utf-8"))
    raw["model"]["input_tokens"]["value"] = 12.0
    with pytest.raises(EfficiencyReceiptError, match="integer|closed alternative"):
        TaskEfficiencyReceipt.from_dict(raw)
    with pytest.raises(EfficiencyReceiptError, match="integer"):
        QuantitativeEvidence.from_dict(
            {
                "status": "measured",
                "value": 12.0,
                "unit": UNIT_TOKENS,
                "sensor_id": sensor_id_for("float"),
            }
        )
    with pytest.raises(EfficiencyReceiptError, match="canonical bytes"):
        TaskEfficiencyReceipt.from_canonical_bytes(
            json.dumps(receipt.to_dict(), indent=2).encode("utf-8")
        )
    with pytest.raises(EfficiencyReceiptError, match="canonical bytes"):
        admit_task_efficiency_receipt(receipt.canonical_bytes() + b"\n")
    with pytest.raises(EfficiencyReceiptError, match="integer"):
        QuantitativeEvidence.measured(
            True,  # noqa: FBT003
            unit=UNIT_COUNT,
            sensor_id=sensor_id_for("bool"),
        )


def test_missing_required_identity_fails_closed() -> None:
    payload = _receipt().to_dict()
    del payload["identity"]["task_id"]
    with pytest.raises(EfficiencyReceiptError, match="missing required"):
        TaskEfficiencyReceipt.from_dict(payload)
    payload = _receipt().to_dict()
    del payload["identity"]
    with pytest.raises(EfficiencyReceiptError, match="missing required"):
        TaskEfficiencyReceipt.from_dict(payload)
    payload = _receipt().to_dict()
    payload["identity"]["task_id"] = ""
    with pytest.raises(EfficiencyReceiptError):
        TaskEfficiencyReceipt.from_dict(payload)


def test_rejects_conflated_truth_states() -> None:
    base = _receipt()
    with pytest.raises(EfficiencyReceiptError, match="conflated"):
        replace(
            base.cost,
            token_based_estimated_charge=QuantitativeEvidence.measured(
                9,
                unit=UNIT_MICROUSD,
                sensor_id=sensor_id_for("cost.token_based_estimated_charge"),
            ),
        )
    observed = QuantitativeEvidence.observed(
        3,
        unit=UNIT_COUNT,
        sensor_id=sensor_id_for("work.tests_executed"),
    )
    verified_shape = observed.to_dict()
    verified_shape["status"] = "verified"
    verified_shape["verifier_identity"] = "pytest"
    verified_shape["verifier_receipt_cid"] = fixture_cid("verifier")
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        QuantitativeEvidence.from_dict(verified_shape)
    attempted = QuantitativeEvidence.attempted("retry", attempt_id="attempt:2")
    observed_shape = attempted.to_dict()
    observed_shape["status"] = "observed"
    observed_shape["value"] = 1
    observed_shape["unit"] = UNIT_COUNT
    observed_shape["sensor_id"] = sensor_id_for("work.retries")
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        QuantitativeEvidence.from_dict(observed_shape)
    unavailable = unavailable_quantitative("model.input_tokens")
    zeroed = dict(unavailable.to_dict())
    zeroed["value"] = 0
    zeroed["unit"] = UNIT_TOKENS
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        QuantitativeEvidence.from_dict(zeroed)
    live = replace(
        base.terminal,
        population_kind=PopulationKind.LIVE,
    )
    simulated_quality = replace(
        base.quality,
        false_positives=QuantitativeEvidence.simulated(
            0,
            unit=UNIT_COUNT,
            fixture_identity="live-false-positives",
        ),
    )
    with pytest.raises(EfficiencyReceiptError, match="simulated"):
        TaskEfficiencyReceipt(
            identity=base.identity,
            model=base.model,
            compute=base.compute,
            cost=base.cost,
            work=base.work,
            quality=simulated_quality,
            terminal=live,
        )


def test_measured_zero_requires_sensor_and_is_not_unavailable() -> None:
    measured_zero = QuantitativeEvidence.measured(
        0,
        unit=UNIT_COUNT,
        sensor_id=sensor_id_for("work.human_interventions", "measured"),
    )
    receipt = _receipt(
        work=replace(_receipt().work, human_interventions=measured_zero)
    )
    payload = receipt.to_dict()["work"]["human_interventions"]
    assert payload["status"] == "measured"
    assert payload["value"] == 0
    assert payload["sensor_id"]
    assert "reason_code" not in payload
    assert "work.human_interventions" in receipt.evidence_state()["measured"]
    assert "work.human_interventions" not in receipt.evidence_state()["unavailable"]


def test_schema_validation_rejects_unknown_and_bounds_on_wire_payloads() -> None:
    schema = load_task_efficiency_receipt_schema()
    payload = _distinguished_receipt().to_dict()
    validate_closed_schema(payload, schema)
    unknown = copy.deepcopy(payload)
    unknown["model"]["input_tokens"]["extra"] = 1
    with pytest.raises(EfficiencyReceiptError, match="unknown fields|closed alternative"):
        validate_closed_schema(unknown, schema)
    bounded = copy.deepcopy(payload)
    bounded["model"]["input_tokens"]["value"] = 10**18 + 1
    with pytest.raises(EfficiencyReceiptError, match="maximum|closed alternative"):
        validate_closed_schema(bounded, schema)


def test_mismatched_evidence_state_fails_closed() -> None:
    payload = _distinguished_receipt().to_dict()
    payload["evidence_state"]["measured"] = []
    payload["evidence_state"]["unavailable"].append("model.input_tokens")
    payload["evidence_state"]["unavailable"] = sorted(
        set(payload["evidence_state"]["unavailable"])
    )
    with pytest.raises(EfficiencyReceiptError, match="evidence_state"):
        TaskEfficiencyReceipt.from_dict(payload)


def test_provider_request_ids_reject_credentials() -> None:
    with pytest.raises(EfficiencyReceiptError, match="credentials"):
        RequestIdEvidence.observed(
            ["api_key-123"],
            sensor_id=sensor_id_for("model.safe_provider_request_ids"),
        )


def test_work_section_rejects_unknown_metric() -> None:
    payload = _receipt().work.to_dict()
    payload["invented"] = payload["retries"]
    with pytest.raises(EfficiencyReceiptError, match="unknown fields"):
        WorkUse.from_dict(payload)
