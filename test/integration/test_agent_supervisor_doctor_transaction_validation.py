"""Validation obligations execute on real candidate bytes before Git CAS."""

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorPlanStep,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transaction import (
    DeterministicDoctorTransaction,
    DeterministicDoctorTransactionError,
    DoctorStepApplyResult,
    DoctorStepDisposition,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_worktree_adapter import DoctorExactEdit
from test.integration.test_agent_supervisor_doctor_transaction_live import (
    _adapter,
    _git,
    _hash,
    _plan,
    _repo,
)


def setup(tmp_path):
    before = {"pkg/a.py": b"value = 1\n"}
    root = _repo(tmp_path, before)
    base = _git(root, "rev-parse", "refs/heads/main")
    plan = _plan(before, one_scc=False)
    validation = DoctorPlanStep(
        step_id="step:validate",
        kind="validation",
        dependency_step_ids=(plan.steps[0].step_id,),
        consumer_ids=(plan.consumer_dispositions[0].consumer_id,),
        validation_refs=("validation:ast",),
    )
    plan = replace(plan, steps=(*plan.steps, validation), validation_refs=("validation:ast",))
    kwargs = {
        "worktree_adapter": _adapter(root, tmp_path / "state", tuple(before)),
        "edits": (DoctorExactEdit("pkg/a.py", _hash(before["pkg/a.py"]), b"value = 2\n"),),
        "target_ref": "refs/heads/main",
        "base_ref": base,
    }
    return root, base, plan, kwargs


def test_validation_requires_callback_before_worktree_or_ref_mutation(tmp_path):
    root, base, plan, kwargs = setup(tmp_path)
    with pytest.raises(DeterministicDoctorTransactionError, match="step_validator"):
        DeterministicDoctorTransaction().execute_live(plan, **kwargs)
    assert _git(root, "rev-parse", "refs/heads/main") == base
    assert not any(path.is_file() for path in (tmp_path / "state").rglob("*"))


def test_successful_validation_reads_candidate_and_is_retained_before_cas(tmp_path):
    import ast

    root, base, plan, kwargs = setup(tmp_path)
    seen = []

    def validate(session, exact_plan, step):
        assert exact_plan == plan
        assert _git(root, "rev-parse", "refs/heads/main") == base
        source = (session.worktree_root / "pkg/a.py").read_bytes()
        assert source == b"value = 2\n"
        ast.parse(source)
        seen.append(step.step_id)
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.PASSED,
            diagnostic_refs=(f"validated:{step.step_id}:{_hash(source)}",),
            static_replay=True,
        )

    report = DeterministicDoctorTransaction().execute_live(plan, **kwargs, step_validator=validate)
    assert report.committed
    assert seen == [step.step_id for step in plan.steps]
    records = [step for group in report.group_receipts for step in group.step_receipts]
    assert all(step.diagnostic_refs[0].startswith("validated:") for step in records)
    assert _git(root, "show", "refs/heads/main:pkg/a.py") == "value = 2"


@pytest.mark.parametrize("outcome", ["failed", "raises", "missing_evidence", "malformed"])
def test_validation_failure_restores_and_cannot_advance_ref(tmp_path, outcome):
    root, base, plan, kwargs = setup(tmp_path)

    def validate(session, exact_plan, step):
        assert (session.worktree_root / "pkg/a.py").read_bytes() == b"value = 2\n"
        if outcome == "raises":
            raise RuntimeError("validator failed")
        if outcome == "malformed":
            return {"passed": True}
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.PASSED
            if outcome == "missing_evidence"
            else DoctorStepDisposition.FAILED,
            diagnostic_refs=()
            if outcome == "missing_evidence"
            else ("validation:failed-on-candidate",),
        )

    report = DeterministicDoctorTransaction().execute_live(plan, **kwargs, step_validator=validate)
    assert not report.committed
    assert _git(root, "rev-parse", "refs/heads/main") == base
    assert (root / "pkg/a.py").read_bytes() == b"value = 1\n"
    if outcome == "failed":
        assert any(
            "validation:failed-on-candidate" in step.diagnostic_refs
            for group in report.group_receipts
            for step in group.step_receipts
        )


def test_validator_cannot_mutate_candidate_before_reporting_success(tmp_path):
    root, base, plan, kwargs = setup(tmp_path)

    def validate(session, exact_plan, step):
        (session.worktree_root / "pkg/a.py").write_bytes(b"value = 999\n")
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.PASSED,
            diagnostic_refs=("validation:claimed-pass",),
            static_replay=True,
        )

    with pytest.raises(DeterministicDoctorTransactionError, match="mutated"):
        DeterministicDoctorTransaction().execute_live(plan, **kwargs, step_validator=validate)
    assert _git(root, "rev-parse", "refs/heads/main") == base
    assert (root / "pkg/a.py").read_bytes() == b"value = 1\n"


def test_validation_gate_executes_before_dependent_write_group(tmp_path):
    before = {"pkg/a.py": b"a = 1\n", "pkg/b.py": b"b = 1\n"}
    root = _repo(tmp_path, before)
    plan = _plan(before, one_scc=False)
    gate = DoctorPlanStep(
        step_id="step:gate",
        kind="validation",
        dependency_step_ids=("step:0",),
        validation_refs=("validation:intermediate",),
    )
    second = replace(plan.steps[1], dependency_step_ids=(gate.step_id,))
    plan = replace(plan, steps=(plan.steps[0], gate, second))
    seen = []

    def validate(session, exact_plan, step):
        a = (session.worktree_root / "pkg/a.py").read_bytes()
        b = (session.worktree_root / "pkg/b.py").read_bytes()
        assert a == b"a = 2\n"
        assert b == (b"b = 2\n" if step.step_id == "step:1" else b"b = 1\n")
        seen.append(step.step_id)
        return DoctorStepApplyResult(
            disposition=DoctorStepDisposition.PASSED,
            diagnostic_refs=("checked:" + step.step_id,),
            static_replay=True,
        )

    result = DeterministicDoctorTransaction().execute_live(
        plan,
        worktree_adapter=_adapter(root, tmp_path / "state", tuple(before)),
        edits=(
            DoctorExactEdit("pkg/a.py", _hash(before["pkg/a.py"]), b"a = 2\n"),
            DoctorExactEdit("pkg/b.py", _hash(before["pkg/b.py"]), b"b = 2\n"),
        ),
        target_ref="refs/heads/main",
        step_validator=validate,
    )
    assert result.committed
    assert seen == ["step:0", "step:gate", "step:1"]
