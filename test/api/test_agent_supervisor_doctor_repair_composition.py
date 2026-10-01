"""Native composition boundaries and a real Lean/Z3/worktree qualification."""

from dataclasses import replace
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition import (
    DoctorCompositionError,
    composition_snapshot,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction import (
    DoctorStepApplyResult,
    DoctorStepDisposition,
)


@pytest.fixture
def smoke():
    if not (os.environ.get("DOCTOR_COMPOSITION_LEAN") or shutil.which("elan")):
        pytest.skip("real Lean toolchain is required")
    if not (os.environ.get("DOCTOR_COMPOSITION_Z3") or shutil.which("z3")):
        pytest.skip("real Z3 is required")
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/agent_supervisor/container_coding/doctor_composition_smoke.py"
    )
    spec = importlib.util.spec_from_file_location("doctor_composition_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def bound(tmp_path, smoke):
    return smoke.prepare(tmp_path / "qualification")


def test_real_provers_synthesis_impact_and_validated_ref_commit(tmp_path, smoke):
    runtime, inputs = smoke.run(tmp_path / "real")
    composed = runtime.composition_result
    assert composed.admitted
    assert composed.proof.mutation_capable
    assert composed.synthesis.authoritative_proof
    assert composed.impact.mutation_admissible
    report = runtime.composition_transaction
    assert report.committed
    assert report.merge_cas.ref_name == inputs.target_ref
    assert report.merge_cas.expected_ref == inputs.base_ref
    assert report.merge_cas.desired_ref != inputs.base_ref
    qualification = json.loads((tmp_path / "real/qualification.json").read_text())
    assert qualification["after_behavior_passed"]
    assert not qualification["task_completion_authorized"]
    assert (runtime.checkout_root / "pkg.py").read_text() == inputs.synthesis.file_text
    assert all(
        step.diagnostic_refs for group in report.group_receipts for step in group.step_receipts
    )


@pytest.mark.parametrize("change", ["impact_subject", "proof_assertion", "consequence", "snapshot"])
def test_cross_bound_or_self_asserted_inputs_rejected(bound, change):
    runtime, inputs = bound
    with pytest.raises(DoctorCompositionError):
        if change == "impact_subject":
            replace(inputs, impact=replace(inputs.impact, subject_symbol_id="unrelated"))
        elif change == "proof_assertion":
            replace(inputs, synthesis=replace(inputs.synthesis, proof_receipt={"admitted": True}))
        elif change == "consequence":
            replace(inputs, expected_after_hash="sha256:" + "f" * 64)
        else:
            composition_snapshot(
                runtime.evidence.snapshot, replace(inputs.snapshot.roots, tree_id="tree:other")
            )


def test_source_drift_rejected_before_prover_or_mutation(bound):
    runtime, inputs = bound
    (runtime.checkout_root / "pkg.py").write_text(inputs.synthesis.file_text + "\n# drift\n")
    with pytest.raises(DoctorCompositionError, match="preimage changed"):
        runtime.plan()
    assert runtime.composition_result is None


def test_omitted_or_opaque_inventory_rejected(bound):
    runtime, inputs = bound
    evidence = runtime.evidence
    row = replace(evidence.source_inventory[0], path="omitted.py")
    expanded = replace(evidence, source_inventory=(*evidence.source_inventory, row))
    with pytest.raises(DoctorCompositionError, match="inventory"):
        replace(inputs, evidence_id=expanded.evidence_id).assert_current(
            runtime.checkout_root, expanded
        )
    opaque = replace(evidence, source_inventory=(replace(row, coverage_kind="structured_data"),))
    with pytest.raises(DoctorCompositionError, match="non-semantic"):
        replace(inputs, evidence_id=opaque.evidence_id).assert_current(
            runtime.checkout_root, opaque
        )


def test_truncated_diagnostic_scope_cannot_close_impact(bound):
    runtime, inputs = bound
    evidence = replace(runtime.evidence, notes=("diagnostic_source_bound_reached",))
    with pytest.raises(DoctorCompositionError, match="bounded diagnostic"):
        replace(inputs, evidence_id=evidence.evidence_id).assert_current(
            runtime.checkout_root, evidence
        )


def test_failed_candidate_validator_rolls_back_before_ref_cas(bound, monkeypatch):
    runtime, inputs = bound
    runtime.plan()
    composed = runtime.composition_result
    assert composed.admitted

    def reject(_composition):
        return lambda *_args: DoctorStepApplyResult(
            disposition=DoctorStepDisposition.FAILED,
            reason_codes=("qualification_validator_rejected",),
            diagnostic_refs=("validation:negative-control",),
            static_replay=True,
        )

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition.composition_step_validator",
        reject,
    )
    result = runtime.execute(
        "repair", plan=composed.plan, mode="sandbox_auto", exact_clean_target=True
    )
    assert not result.result.changed
    assert not runtime.composition_transaction.committed
    current = subprocess.check_output(
        ["git", "-C", str(runtime.checkout_root), "rev-parse", inputs.target_ref],
        text=True,
    ).strip()
    assert current == inputs.base_ref
    assert (runtime.checkout_root / "pkg.py").read_text() == inputs.synthesis.file_text
