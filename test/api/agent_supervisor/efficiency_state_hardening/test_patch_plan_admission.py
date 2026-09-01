"""ASEH-055: PatchPlan construction and fail-closed merge admission."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.patch_admission import (
    PatchAdmissionDisposition,
    ValidationReceipt,
    admit_patch_plan,
)
from ipfs_accelerate_py.agent_supervisor.merge.patch_plan import (
    FileSymbol,
    PatchPlan,
    PatchPlanError,
    patch_digest,
)


BASE_TREE = "f0b02bfd2bf2be2ca7ee24c1576786ccd86b0271"
PATCH = """diff --git a/pkg/worker.py b/pkg/worker.py
index 1111111..2222222 100644
--- a/pkg/worker.py
+++ b/pkg/worker.py
@@ -1 +1 @@
-return old_value
+return new_value
"""


def _plan(patch: str = PATCH, **overrides: object) -> PatchPlan:
    fields: dict[str, object] = {
        "base_tree": BASE_TREE,
        "intended_semantic_change": "Return the corrected worker result.",
        "files_and_symbols": (FileSymbol("pkg/worker.py", ("resolve_worker",)),),
        "preconditions": ("The worker contract remains available.",),
        "postconditions": ("The corrected result is returned.",),
        "expected_invariants": ("The public worker interface is unchanged.",),
        "required_tests_and_proofs": ("test_worker_result",),
        "scope_limit": ("pkg",),
        "patch_digest": patch_digest(patch),
    }
    fields.update(overrides)
    return PatchPlan(**fields)  # type: ignore[arg-type]


def _receipt(plan: PatchPlan, **overrides: object) -> ValidationReceipt:
    fields: dict[str, object] = {
        "receipt_id": "validation-001",
        "tree_id": BASE_TREE,
        "patch_digest": plan.patch_digest,
        "plan_digest": plan.plan_digest,
        "passed": True,
    }
    fields.update(overrides)
    return ValidationReceipt(**fields)  # type: ignore[arg-type]


def _admit(plan: PatchPlan, patch: str = PATCH, **kwargs: object):
    validation_receipt = kwargs.pop("validation_receipt", _receipt(plan))
    current_tree = kwargs.pop("current_tree", BASE_TREE)
    return admit_patch_plan(
        plan,
        patch=patch,
        current_tree=current_tree,  # type: ignore[arg-type]
        validation_receipt=validation_receipt,  # type: ignore[arg-type]
        **kwargs,
    )


def test_closed_patch_plan_round_trip_and_schema_are_canonical() -> None:
    plan = _plan()
    reconstructed = PatchPlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    schema_path = Path("ipfs_accelerate_py/agent_supervisor/merge/schemas/patch_plan.schema.json")
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert reconstructed == plan
    assert reconstructed.plan_digest == plan.plan_digest
    assert schema["properties"]["schema"]["const"] == plan.schema
    assert set(schema["required"]) == set(plan.to_dict())


def test_admission_binds_digest_tree_scope_and_current_receipt() -> None:
    plan = _plan()
    receipt = _admit(plan)

    assert receipt.disposition is PatchAdmissionDisposition.ADMITTED
    assert receipt.admitted
    assert receipt.plan_digest == plan.plan_digest
    assert receipt.patch_digest == plan.patch_digest
    assert receipt.validation_receipt_id == "validation-001"
    assert receipt.reason_codes == ()


def test_admission_rejects_nonempty_patch_with_digest_or_scope_mismatch() -> None:
    plan = _plan()
    wrong_digest = _admit(plan, PATCH.replace("new_value", "different_value"))
    out_of_scope_patch = PATCH.replace("pkg/worker.py", "other/worker.py")
    out_of_scope = _admit(
        _plan(out_of_scope_patch), out_of_scope_patch,
        validation_receipt=_receipt(_plan(out_of_scope_patch)),
    )

    assert "patch_digest_mismatch" in wrong_digest.reason_codes
    assert "out_of_scope_path" in out_of_scope.reason_codes
    assert "files_and_symbols_mismatch" in out_of_scope.reason_codes


@pytest.mark.parametrize("patch", ["", "diff --git a/pkg/worker.py b/pkg/worker.py\n--- a/pkg/worker.py\n+++ b/pkg/worker.py\n"])
def test_admission_rejects_empty_patch(patch: str) -> None:
    plan = _plan(patch)
    receipt = _admit(plan, patch)

    assert receipt.disposition is PatchAdmissionDisposition.REJECTED
    assert "semantically_empty_patch" in receipt.reason_codes


@pytest.mark.parametrize(
    ("patch", "reason"),
    [
        (PATCH.replace("new_value", "api_key = 'super-secret-value-123'"), "secret_change_forbidden"),
        (PATCH.replace("pkg/worker.py", "build/worker.py"), "generated_artifact_forbidden"),
    ],
)
def test_admission_rejects_secret_or_generated_artifact(patch: str, reason: str) -> None:
    plan = _plan(patch, files_and_symbols=(FileSymbol("build/worker.py" if "build/" in patch else "pkg/worker.py", ("resolve_worker",)),), scope_limit=("build" if "build/" in patch else "pkg",))
    receipt = _admit(plan, patch, validation_receipt=_receipt(plan))

    assert receipt.disposition is PatchAdmissionDisposition.REJECTED
    assert reason in receipt.reason_codes


def test_admission_requires_current_passing_validation_receipt() -> None:
    plan = _plan()
    missing = admit_patch_plan(plan, patch=PATCH, current_tree=BASE_TREE, validation_receipt=None)
    failed = _admit(plan, validation_receipt=_receipt(plan, passed=False))
    stale = _admit(plan, validation_receipt=_receipt(plan, tree_id="new-current-tree"))

    assert "current_validation_receipt_missing" in missing.reason_codes
    assert "validation_failed" in failed.reason_codes
    assert "stale_validation_receipt" in stale.reason_codes


def test_admission_rejects_conflict_and_stale_plan() -> None:
    plan = _plan()
    conflict = _admit(plan, conflict_paths=("pkg/worker.py",))
    stale = admit_patch_plan(
        plan, patch=PATCH, current_tree="new-current-tree", validation_receipt=_receipt(plan, tree_id="new-current-tree")
    )

    assert "merge_conflict_detected" in conflict.reason_codes
    assert "stale_base_tree" in stale.reason_codes


def test_patch_plan_rejects_unbound_or_unsafe_contract_fields() -> None:
    with pytest.raises(PatchPlanError, match="scope_limit"):
        _plan(scope_limit=("../escape",))
    with pytest.raises(PatchPlanError, match="patch_digest"):
        _plan(patch_digest="not-a-digest")
    with pytest.raises(PatchPlanError, match="preconditions"):
        _plan(preconditions=())
