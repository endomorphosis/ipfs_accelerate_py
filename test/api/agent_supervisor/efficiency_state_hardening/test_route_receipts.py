"""ASEH-024 closed route receipts and escalation metrics."""

from __future__ import annotations

import json
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.decision_receipts import (
    AUTHORITY,
    DETERMINISTIC_STAGES,
    DecisionReceiptError,
    EVIDENCE_KINDS,
    EXECUTOR_CLASSES,
    IDENTITY_FIELDS,
    ORDERED_STAGES,
    REQUESTED_RATES,
    RESOURCE_QUANTITY_FIELDS,
    ROUTE_RECEIPT_FIELDS,
    ROUTING_DECISION_SCHEMA,
    ROUTING_DECISION_SCHEMA_PATH,
    admit_route_receipt,
    build_route_receipt,
    canonical_bytes,
    collect_unavailable_fields,
    compute_requested_rates,
    content_identity,
    emit_decision,
    load_routing_decision_schema,
    measured_quantity,
    measured_rate,
    observed_identity,
    ratio_millionths,
    reconcile_actual_resource_use,
    round_trip_route_receipt,
    selected_executor_for_stage,
    unavailable,
    unavailable_rate,
)


COMMIT = "755f45475cc2d13dacd8b330036c1d597afeddde"
TREE = "729da9f8293ecfa046a0136381a3d3808f9ed140"
POLICY = "ipfs_accelerate_py/agent-supervisor/aseh-supervisor-policy@1"
OBJECTIVE_REVISION = "baguqeeray6iu7h6kiajjow44w3l6gexkiihui2423wrhtpou22thcm6sqhbq"
TASK_CID = content_identity({"task": "ASEH-024"})
QUESTION_ID = "sha256:" + ("ab" * 32)
EVIDENCE_DIGEST = "sha256:" + ("cd" * 32)
CLOSURE_CID = content_identity({"question": "closed"})


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


def _run(stage: str, reason: str, *, resolved: bool) -> dict[str, Any]:
    return {"stage": stage, "reason": reason, "resolved": resolved}


def _skip(stage: str, reason: str) -> dict[str, Any]:
    return {"stage": stage, "reason": reason}


def _partition(
    resolver: str,
    *,
    resolver_reason: str,
    prior_run_reason: str = "unresolved_deterministic_evidence",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    run: list[dict[str, Any]] = []
    skip: list[dict[str, Any]] = []
    seen = False
    for stage in ORDERED_STAGES:
        if stage == resolver:
            run.append(_run(stage, resolver_reason, resolved=True))
            seen = True
            continue
        if seen:
            skip.append(_skip(stage, "resolved_by_prior_stage"))
            continue
        if stage in DETERMINISTIC_STAGES:
            run.append(_run(stage, prior_run_reason, resolved=False))
        elif resolver == "human_decision" and stage in {
            "local_small_specialist_model",
            "local_or_remote_medium_model",
            "remote_strong_or_frontier_model",
        }:
            skip.append(_skip(stage, "human_gate_preempts_model"))
        elif stage != resolver:
            skip.append(_skip(stage, "not_smallest_adequate_executor"))
    return run, skip


def _evidence(stage: str, *, verified: bool = False) -> dict[str, Any]:
    kind = {
        "exact_current_authoritative_cached_receipt": "cached_receipt",
        "ast_symbol_dependency_and_impact_analysis": "impact_analysis",
        "schema_type_static_lint_and_contract_checks": "static_contract",
        "selected_tests": "selected_tests",
        "incremental_smt_or_theorem_prover": "incremental_proof",
        "local_small_specialist_model": "model_answer",
        "local_or_remote_medium_model": "model_answer",
        "remote_strong_or_frontier_model": "model_answer",
        "human_decision": "human_decision",
    }[stage]
    payload: dict[str, Any] = {
        "stage": stage,
        "evidence_kind": kind,
        "evidence_id": EVIDENCE_DIGEST,
        "truth_state": "verified" if verified else "observed",
        "summary": f"decisive {kind}",
    }
    if verified:
        payload["verifier_id"] = "aseh-024-verifier"
        payload["verifier_receipt_cid"] = CLOSURE_CID
    return payload


def _question(*, status: str = "closed") -> dict[str, Any]:
    payload: dict[str, Any] = {
        "status": status,
        "question_id": observed_identity(QUESTION_ID),
    }
    if status == "closed":
        payload["closure_evidence"] = observed_identity(CLOSURE_CID)
    elif status == "open":
        payload["closure_evidence"] = unavailable("not_yet_measured")
    else:
        payload["question_id"] = unavailable("not_applicable")
        payload["closure_evidence"] = unavailable("not_applicable")
    return payload


def _measured(value: int, *, unit: str, sensor: str) -> dict[str, Any]:
    return measured_quantity(value, unit=unit, sensor_id=sensor)


def _provider_usage(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "admitted": True,
        "task_id": "ASEH-024",
        "span_id": "span:task-aseh-024",
        "record_id": "usage:aseh-024",
        "input_tokens": _measured(120, unit="tokens", sensor="provider:grok"),
        "output_tokens": _measured(40, unit="tokens", sensor="provider:grok"),
        "cached_input_tokens": _measured(16, unit="tokens", sensor="provider:grok"),
        "reasoning_tokens": _measured(8, unit="tokens", sensor="provider:grok"),
        "number_of_calls": _measured(1, unit="count", sensor="provider:grok"),
        "provider_reported_charge": _measured(2500, unit="microusd", sensor="provider:grok"),
    }
    payload.update(overrides)
    return payload


def _work_telemetry(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "admitted": True,
        "task_id": "ASEH-024",
        "span_id": "span:task-aseh-024",
        "record_id": "work:aseh-024",
        "final_task_outcome": "succeeded",
        "cpu_seconds": _measured(3_000_000, unit="seconds_millionths", sensor="procfs"),
        "gpu_seconds": unavailable("hardware_absent"),
        "peak_memory": _measured(96_000_000, unit="bytes", sensor="procfs"),
        "wall_clock_duration": _measured(4_000_000, unit="seconds_millionths", sensor="mono"),
        "audit_and_verification_overhead": _measured(
            250, unit="microusd", sensor="sensor:audit-runner"
        ),
    }
    payload.update(overrides)
    return payload


def _receipt(**overrides: Any) -> Any:
    resolver = str(overrides.pop("resolver", "selected_tests"))
    reason_map = {
        "selected_tests": "eligible_deterministic_evidence",
        "local_small_specialist_model": "smallest_adequate_executor",
        "local_or_remote_medium_model": "smallest_adequate_executor",
        "remote_strong_or_frontier_model": "smallest_adequate_executor",
        "human_decision": "human_decision_required",
    }
    escalation_map = {
        "selected_tests": "not_applicable",
        "local_small_specialist_model": "smallest_adequate_executor",
        "local_or_remote_medium_model": "smallest_adequate_executor",
        "remote_strong_or_frontier_model": "smallest_adequate_executor",
        "human_decision": "human_decision_required",
    }
    run, skip = _partition(resolver, resolver_reason=reason_map[resolver])
    kwargs: dict[str, Any] = {
        "task_id": "ASEH-024",
        "task_cid": TASK_CID,
        "objective_id": "ASEH-G030",
        "objective_revision": OBJECTIVE_REVISION,
        "repository_commit": COMMIT,
        "repository_tree": TREE,
        "policy_identity": POLICY,
        "stages_run": run,
        "stages_skipped_and_reason": skip,
        "decisive_evidence": _evidence(resolver),
        "escalation_reason": escalation_map[resolver],
        "final_outcome": "succeeded",
        "work_telemetry": _work_telemetry(),
    }
    if resolver in DETERMINISTIC_STAGES:
        kwargs["unnecessary_escalation"] = False
    else:
        kwargs["question_closure"] = _question(status="closed")
        kwargs["unnecessary_escalation"] = False
        kwargs["provider_usage"] = _provider_usage()
    kwargs.update(overrides)
    return build_route_receipt(**kwargs)


def test_schema_file_exists_and_is_closed() -> None:
    schema = load_routing_decision_schema()
    assert ROUTING_DECISION_SCHEMA_PATH.is_file()
    assert schema["$id"] == ROUTING_DECISION_SCHEMA
    assert _walk_object_schemas(schema) == []
    assert tuple(schema["$defs"]["identity"]["required"]) == IDENTITY_FIELDS
    assert tuple(schema["$defs"]["metrics"]["required"]) == REQUESTED_RATES
    for field in ROUTE_RECEIPT_FIELDS:
        assert field in schema["required"]
        assert field in schema["properties"]
    assert set(schema["$defs"]["ladderStage"]["enum"]) == set(ORDERED_STAGES)
    assert set(schema["$defs"]["evidenceKind"]["enum"]) == set(EVIDENCE_KINDS)
    assert set(schema["$defs"]["executorClass"]["enum"]) == set(EXECUTOR_CLASSES)


def test_emit_decision_remains_typed() -> None:
    envelope = emit_decision({"decision": "select", "metrics": {"ok": 1}})
    assert envelope["schema"] == "lgswf/decision-receipt@1"
    assert envelope["decision"] == "select"
    assert envelope["metrics"] == {"ok": 1}


def test_receipt_round_trips_canonical_bytes() -> None:
    receipt = _receipt()
    encoded = receipt.canonical_bytes()
    assert encoded == canonical_bytes(json.loads(encoded.decode("utf-8")))
    replayed = admit_route_receipt(encoded)
    assert replayed.canonical_bytes() == encoded
    assert replayed.receipt_cid == receipt.receipt_cid
    assert round_trip_route_receipt(encoded) == encoded
    pretty = json.dumps(json.loads(encoded.decode("utf-8")), indent=2)
    assert admit_route_receipt(pretty).canonical_bytes() == encoded


def test_closed_receipt_records_required_observation_fields() -> None:
    receipt = _receipt()
    payload = receipt.to_dict()
    for field in ROUTE_RECEIPT_FIELDS:
        assert field in payload
    assert payload["stages_run"][-1]["stage"] == "selected_tests"
    assert payload["stages_run"][-1]["resolved"] is True
    skipped = {item["stage"]: item["reason"] for item in payload["stages_skipped_and_reason"]}
    assert skipped["human_decision"] == "resolved_by_prior_stage"
    assert payload["decisive_evidence"]["evidence_kind"] == "selected_tests"
    assert payload["escalation_reason"] == "not_applicable"
    assert payload["selected_executor"] == selected_executor_for_stage("selected_tests")
    assert payload["final_outcome"] == "succeeded"
    assert receipt.validates_patches is False
    assert receipt.promotes_policy is False
    assert receipt.authority == AUTHORITY


def test_reconciles_to_admitted_provider_and_work_telemetry() -> None:
    provider = _provider_usage()
    work = _work_telemetry()
    receipt = _receipt(
        resolver="local_small_specialist_model",
        provider_usage=provider,
        work_telemetry=work,
    )
    resources = receipt.actual_resource_use
    assert resources["input_tokens"]["value"] == 120
    assert resources["output_tokens"]["value"] == 40
    assert resources["cpu_seconds"]["value"] == 3_000_000
    assert resources["gpu_seconds"]["truth_state"] == "unavailable"
    assert resources["provider_usage_record_id"]["value"] == "usage:aseh-024"
    assert resources["work_telemetry_record_id"]["value"] == "work:aseh-024"
    assert resources["span_id"]["value"] == "span:task-aseh-024"
    projected = reconcile_actual_resource_use(
        task_id="ASEH-024",
        actual_resource_use=resources,
        provider_usage=provider,
        work_telemetry=work,
    )
    for name in RESOURCE_QUANTITY_FIELDS:
        assert projected[name] == resources[name]


def test_unbound_usage_rejects_the_receipt() -> None:
    with pytest.raises(DecisionReceiptError, match="unbound usage"):
        _receipt(
            resolver="local_small_specialist_model",
            provider_usage=_provider_usage(admitted=False, task_id="", span_id=""),
        )
    with pytest.raises(DecisionReceiptError, match="unbound usage"):
        _receipt(
            resolver="local_small_specialist_model",
            provider_usage=_provider_usage(task_id="OTHER"),
        )
    with pytest.raises(DecisionReceiptError, match="unbound usage"):
        _receipt(
            work_telemetry=_work_telemetry(admitted=False, span_id=""),
        )


def test_stage_mismatch_rejects_the_receipt() -> None:
    run, skip = _partition(
        "selected_tests", resolver_reason="eligible_deterministic_evidence"
    )
    skip = [item for item in skip if item["stage"] != "human_decision"]
    with pytest.raises(DecisionReceiptError, match="stage mismatch"):
        _receipt(stages_run=run, stages_skipped_and_reason=skip)
    with pytest.raises(DecisionReceiptError, match="stage mismatch"):
        _receipt(
            selected_executor=selected_executor_for_stage(
                "remote_strong_or_frontier_model"
            )
        )


def test_absent_decisive_evidence_rejects_the_receipt() -> None:
    evidence = _evidence("selected_tests")
    evidence["evidence_id"] = "   "
    with pytest.raises(DecisionReceiptError, match="absent decisive evidence"):
        _receipt(decisive_evidence=evidence)
    mismatch = _evidence("selected_tests")
    mismatch["stage"] = "human_decision"
    with pytest.raises(DecisionReceiptError, match="absent decisive evidence"):
        _receipt(decisive_evidence=mismatch)


def test_claimed_closure_without_evidence_rejects_the_receipt() -> None:
    with pytest.raises(DecisionReceiptError, match="claimed closure without evidence"):
        _receipt(
            resolver="local_small_specialist_model",
            question_closure={
                "status": "closed",
                "question_id": observed_identity(QUESTION_ID),
                "closure_evidence": unavailable("not_yet_measured"),
            },
        )


def test_model_route_without_question_fails_closed() -> None:
    with pytest.raises(DecisionReceiptError, match="unresolved question"):
        _receipt(
            resolver="local_small_specialist_model",
            question_closure=_question(status="not_applicable"),
        )


def test_every_requested_rate_is_computed_for_a_closed_receipt() -> None:
    receipt = _receipt()
    rates = receipt.requested_rates()
    assert set(rates) == set(REQUESTED_RATES)
    assert rates["deterministic_resolution_rate"] == measured_rate(1, 1)
    assert rates["small_model_resolution_rate"]["value"] == 0
    assert rates["frontier_model_rate"]["value"] == 0
    assert rates["human_rate"]["value"] == 0
    assert rates["escalation_precision"] == unavailable_rate("not_applicable")
    assert rates["unnecessary_escalation_rate"] == unavailable_rate("not_applicable")
    assert rates["unresolved_question_closure_rate"] == unavailable_rate("not_applicable")
    assert "value" not in rates["escalation_precision"]
    assert "denominator" not in rates["escalation_precision"]


def test_rates_do_not_impute_unavailable_denominators() -> None:
    empty = compute_requested_rates([])
    for name in REQUESTED_RATES:
        assert empty[name]["truth_state"] == "unavailable"
        assert "value" not in empty[name]
        assert empty[name]["reason_code"] == "not_yet_measured"
    assert ratio_millionths(1, 0) == unavailable_rate("not_applicable")
    assert ratio_millionths(1, None) == unavailable_rate("not_yet_measured")
    with pytest.raises(DecisionReceiptError, match="division by missing data"):
        measured_rate(0, 0)
    model = _receipt(
        resolver="local_small_specialist_model",
        unnecessary_escalation=unavailable("not_yet_measured"),
        question_closure=_question(status="open"),
    )
    rates = model.metrics
    assert rates["escalation_precision"]["truth_state"] == "unavailable"
    assert rates["unnecessary_escalation_rate"]["truth_state"] == "unavailable"
    assert "value" not in rates["escalation_precision"]
    assert rates["unresolved_question_closure_rate"] == measured_rate(0, 1)


def test_imputed_zero_denominator_is_rejected_on_admit() -> None:
    payload = _receipt().to_dict()
    payload["metrics"]["escalation_precision"] = {
        "truth_state": "measured",
        "value": 0,
        "unit": "ratio_millionths",
        "numerator": 0,
        "denominator": 1,
        "sensor_id": "aseh-024-route-receipt-rates",
    }
    payload["explicit_unavailable_fields"] = [
        path
        for path in payload["explicit_unavailable_fields"]
        if path != "metrics.escalation_precision"
    ]
    with pytest.raises(DecisionReceiptError, match="division by missing data|does not match"):
        admit_route_receipt(payload)
    payload = _receipt().to_dict()
    payload["metrics"]["escalation_precision"] = {
        "truth_state": "unavailable",
        "reason_code": "not_applicable",
        "value": 0,
        "denominator": 0,
    }
    with pytest.raises(DecisionReceiptError):
        admit_route_receipt(payload)


def test_cohort_rates_cover_deterministic_small_frontier_and_human_shares() -> None:
    receipts = [
        _receipt(),
        _receipt(resolver="local_small_specialist_model"),
        _receipt(resolver="remote_strong_or_frontier_model"),
        _receipt(resolver="human_decision"),
    ]
    rates = compute_requested_rates(receipts)
    assert rates["deterministic_resolution_rate"] == measured_rate(1, 4)
    assert rates["small_model_resolution_rate"] == measured_rate(1, 4)
    assert rates["frontier_model_rate"] == measured_rate(1, 4)
    assert rates["human_rate"] == measured_rate(1, 4)
    assert rates["escalation_precision"] == measured_rate(3, 3)
    assert rates["unnecessary_escalation_rate"] == measured_rate(0, 3)
    assert rates["unresolved_question_closure_rate"] == measured_rate(3, 3)


def test_medium_model_is_not_imputed_into_frontier_or_small_shares() -> None:
    receipt = _receipt(resolver="local_or_remote_medium_model")
    rates = receipt.metrics
    assert rates["deterministic_resolution_rate"]["value"] == 0
    assert rates["small_model_resolution_rate"]["value"] == 0
    assert rates["frontier_model_rate"]["value"] == 0
    assert rates["human_rate"]["value"] == 0
    mixed = compute_requested_rates([_receipt(), receipt])
    assert mixed["deterministic_resolution_rate"] == measured_rate(1, 2)
    assert mixed["frontier_model_rate"] == measured_rate(0, 2)
    assert mixed["small_model_resolution_rate"] == measured_rate(0, 2)


def test_unnecessary_escalation_and_open_question_rates() -> None:
    unnecessary = _receipt(
        resolver="remote_strong_or_frontier_model",
        unnecessary_escalation=True,
        question_closure=_question(status="open"),
    )
    necessary = _receipt(resolver="local_small_specialist_model")
    rates = compute_requested_rates([unnecessary, necessary])
    assert rates["escalation_precision"] == measured_rate(1, 2)
    assert rates["unnecessary_escalation_rate"] == measured_rate(1, 2)
    assert rates["unresolved_question_closure_rate"] == measured_rate(1, 2)
    assert unnecessary.question_closure["status"] == "open"
    assert unnecessary.unnecessary_escalation["unnecessary"] is True


def test_unavailable_escalation_classification_does_not_shrink_the_denominator() -> None:
    known = _receipt(resolver="local_small_specialist_model")
    unknown = _receipt(
        resolver="remote_strong_or_frontier_model",
        unnecessary_escalation=unavailable("not_yet_measured"),
    )
    rates = compute_requested_rates([known, unknown])
    assert rates["escalation_precision"]["truth_state"] == "unavailable"
    assert rates["unnecessary_escalation_rate"]["truth_state"] == "unavailable"
    assert "value" not in rates["escalation_precision"]
    assert rates["frontier_model_rate"] == measured_rate(1, 2)


def test_missing_resources_remain_unavailable_and_never_zero() -> None:
    receipt = _receipt(work_telemetry=None)
    unavailable_paths = receipt.explicit_unavailable_fields
    assert "actual_resource_use.cpu_seconds" in unavailable_paths
    assert "actual_resource_use.input_tokens" in unavailable_paths
    payload = receipt.to_dict()
    for path, node, state in _iter(payload):
        if state == "unavailable":
            assert "value" not in node
            assert "count" not in node
            assert "numerator" not in node
            assert "denominator" not in node
            assert node.get("reason_code")
            assert all(type(child) is not int or child != 0 for child in node.values())
    assert collect_unavailable_fields(payload) == receipt.explicit_unavailable_fields


def _iter(payload: Any):
    from ipfs_accelerate_py.agent_supervisor.runtime.decision_receipts import (
        iter_evidence_nodes,
    )

    return iter_evidence_nodes(payload)


def test_unknown_fields_and_floats_fail_closed() -> None:
    payload = _receipt().to_dict()
    payload["unexpected"] = 1
    with pytest.raises(DecisionReceiptError, match="unknown fields"):
        admit_route_receipt(payload)
    nested = _receipt().to_dict()
    nested["actual_resource_use"]["cpu_seconds"]["extra"] = "nope"
    with pytest.raises(DecisionReceiptError):
        admit_route_receipt(nested)
    floated = _receipt().to_dict()
    floated["actual_resource_use"]["cpu_seconds"]["value"] = 1.5
    with pytest.raises(DecisionReceiptError):
        admit_route_receipt(floated)


def test_receipt_cid_mismatch_and_unavailable_index_fail_closed() -> None:
    payload = _receipt().to_dict()
    payload["receipt_cid"] = CLOSURE_CID
    with pytest.raises(DecisionReceiptError, match="content identity"):
        admit_route_receipt(payload)
    payload = _receipt().to_dict()
    payload["explicit_unavailable_fields"] = []
    with pytest.raises(DecisionReceiptError, match="explicit_unavailable_fields"):
        admit_route_receipt(payload)


def test_work_outcome_must_reconcile() -> None:
    with pytest.raises(DecisionReceiptError, match="work telemetry"):
        _receipt(final_outcome="failed", work_telemetry=_work_telemetry())


def test_authority_cannot_validate_patches_or_promote_policy() -> None:
    payload = _receipt().to_dict()
    payload["authority"] = {
        "observation_only": True,
        "validates_patches": True,
        "promotes_policy": False,
        "schema_attests_measurement": False,
    }
    with pytest.raises(DecisionReceiptError, match="validate patches|promote policy|const"):
        admit_route_receipt(payload)
