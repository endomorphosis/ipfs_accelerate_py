"""ASEH-010 canonical receipt and paired-manifest contract checks."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.efficiency_receipts import (
    EfficiencyReceiptValidationError,
    PairedBenchmarkManifest,
    TaskEfficiencyReceipt,
    TelemetryFact,
    TruthState,
)


ROOT = Path(__file__).resolve().parents[4]
SCHEMAS = ROOT / "ipfs_accelerate_py/agent_supervisor/runtime/schemas"


def _unavailable(reason: str = "not-reported") -> dict[str, str]:
    return {"state": "unavailable", "reason_code": reason}


def _verified(value: object, unit: str = "count") -> dict[str, object]:
    return {
        "state": "verified",
        "value": value,
        "unit": unit if isinstance(value, int) and not isinstance(value, bool) else "",
        "evidence_ref": "receipt:measurement",
        "verifier_ref": "receipt:verification",
    }


def _receipt() -> TaskEfficiencyReceipt:
    identity = {
        name: f"{name}-identity"
        for name in (
            "task_id", "attempt_id", "goal_id", "repository", "repository_commit",
            "repository_tree", "objective_revision", "policy_revision", "environment_id",
            "input_digest",
        )
    }
    model = {
        name: _unavailable()
        for name in (
            "provider", "model_id", "model_revision", "tokenizer_revision", "request_id",
            "input_tokens", "output_tokens", "cached_input_tokens", "reasoning_tokens", "call_count",
        )
    }
    compute = {
        name: _unavailable()
        for name in ("wall_clock_ms", "cpu_ms", "peak_rss_bytes", "gpu_ms", "read_bytes", "write_bytes", "network_bytes")
    }
    cost = {name: _unavailable() for name in ("reported_cost_microusd", "estimated_cost_microusd", "price_snapshot_id", "currency")}
    work = {
        name: _unavailable()
        for name in ("deterministic_steps", "model_calls", "validation_runs", "proof_runs", "retries", "merge_attempts", "human_interventions", "changed_files")
    }
    quality = {
        "acceptance_criteria_total": _verified(1),
        "acceptance_criteria_passed": _verified(0),
        "acceptance_criteria_failed": _verified(1),
        "validation_verdict": _verified("failed"),
        "proof_verdict": _unavailable(),
    }
    return TaskEfficiencyReceipt(
        identity=identity, model=model, compute=compute, cost=cost, work=work, quality=quality,
        evidence={
            "state": "observed", "receipt_references": ["receipt:attempt"],
            "raw_log_references": ["log:bounded"], "validator": _verified("failed"),
            "verification": _unavailable(),
        },
        terminal={"outcome": "failed", "state": "observed", "reason_code": "", "terminal_receipt_ref": "receipt:terminal"},
    )


def test_receipt_round_trips_exact_canonical_bytes() -> None:
    receipt = _receipt()
    canonical = receipt.canonical_bytes()
    assert TaskEfficiencyReceipt.from_canonical_bytes(canonical) == receipt
    assert TaskEfficiencyReceipt.from_dict(receipt.to_dict()).canonical_bytes() == canonical
    assert canonical == receipt.canonical_bytes()
    assert b" " not in canonical


def test_closed_schema_and_bounds_reject_before_admission() -> None:
    payload = _receipt().to_dict()
    unknown = copy.deepcopy(payload)
    unknown["untrusted_extension"] = True
    with pytest.raises(EfficiencyReceiptValidationError):
        TaskEfficiencyReceipt.from_dict(unknown)

    out_of_bound = copy.deepcopy(payload)
    out_of_bound["compute"]["cpu_ms"] = {
        "state": "measured", "value": 10**18 + 1, "unit": "ms", "evidence_ref": "sensor:cpu",
    }
    with pytest.raises(EfficiencyReceiptValidationError):
        TaskEfficiencyReceipt.from_dict(out_of_bound)

    noncanonical = _receipt().canonical_bytes().replace(b"\",\"", b"\", \"")
    with pytest.raises(EfficiencyReceiptValidationError):
        TaskEfficiencyReceipt.from_canonical_bytes(noncanonical)


def test_truth_states_are_distinct_and_have_noninterchangeable_requirements() -> None:
    assert {state.value for state in TruthState} == {
        "measured", "estimated", "simulated", "unavailable", "attempted", "observed", "verified",
    }
    assert TelemetryFact("unavailable", reason_code="sensor-absent").state is TruthState.UNAVAILABLE
    assert TelemetryFact("attempted", evidence_ref="attempt:1").state is TruthState.ATTEMPTED
    assert TelemetryFact("estimated", value=3, unit="tokens", basis_ref="price:1").state is TruthState.ESTIMATED
    assert TelemetryFact("simulated", value=3, unit="tokens", basis_ref="fixture:1").state is TruthState.SIMULATED
    assert TelemetryFact("measured", value=3, unit="tokens", evidence_ref="sensor:1").state is TruthState.MEASURED
    assert TelemetryFact("observed", value=3, unit="tokens", evidence_ref="log:1").state is TruthState.OBSERVED
    assert TelemetryFact("verified", value=3, unit="tokens", evidence_ref="log:1", verifier_ref="verify:1").state is TruthState.VERIFIED
    with pytest.raises(EfficiencyReceiptValidationError):
        TelemetryFact("estimated", value=3, unit="tokens", evidence_ref="sensor:1")
    with pytest.raises(EfficiencyReceiptValidationError):
        TelemetryFact("unavailable", value=0, unit="tokens", reason_code="missing")


def test_paired_manifest_requires_equal_inputs_gates_and_distinct_arms() -> None:
    arms = {
        name: {"receipt_id": f"receipt:{name}", "input_digest": "input:1", "gate_digest": "gate:1", "state": "observed"}
        for name in ("direct", "sealed_current", "candidate")
    }
    manifest = PairedBenchmarkManifest("benchmark:1", "cohort:1", "input:1", "gate:1", arms, "observed")
    assert PairedBenchmarkManifest.from_canonical_bytes(manifest.canonical_bytes()) == manifest
    bad = copy.deepcopy(manifest.to_dict())
    bad["arms"]["candidate"]["input_digest"] = "input:other"
    with pytest.raises(EfficiencyReceiptValidationError):
        PairedBenchmarkManifest.from_dict(bad)


def test_machine_readable_schemas_are_closed_and_parseable() -> None:
    receipt_schema = json.loads((SCHEMAS / "task_efficiency_receipt.schema.json").read_text())
    manifest_schema = json.loads((SCHEMAS / "paired_benchmark_manifest.schema.json").read_text())
    assert receipt_schema["additionalProperties"] is False
    assert manifest_schema["additionalProperties"] is False
    assert set(receipt_schema["$defs"]["truthState"]["enum"]) == {state.value for state in TruthState}
