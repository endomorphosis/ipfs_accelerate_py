"""ASEH-055: PatchPlan plan/receipt admission remains closed and fail-closed."""

from __future__ import annotations

import copy

import pytest

from ipfs_accelerate_py.agent_supervisor.merge.patch_admission import (
    PatchAdmission,
    ValidationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.merge.patch_plan import PatchPlan, PatchPlanValidationError, PatchScope


TREE = "tree:aseh-055"
NOW = 1_700_000_000
PATCH = """diff --git a/pkg/example.py b/pkg/example.py
index 0000000..1111111 100644
--- a/pkg/example.py
+++ b/pkg/example.py
@@ -1 +1 @@
-old = 1
+new = 2
"""


def _plan(patch: str = PATCH) -> PatchPlan:
    return PatchPlan.create(
        patch_text=patch, plan_id="plan:aseh-055", base_tree=TREE,
        semantic_intent="replace the bounded example behavior", files=("pkg/example.py",),
        symbols=("pkg.example.new",), preconditions=("base source is visible",),
        postconditions=("new behavior is active",), invariants=("scope remains bounded",),
        tests=("pytest test/pkg/test_example.py",), proofs=("proof:example",),
        scope=PatchScope(("pkg/**",)),
    )


def _receipt(plan: PatchPlan, *, passed: bool = True, tree: str = TREE, issued_at: int = NOW, independent: bool = True) -> ValidationReceipt:
    return ValidationReceipt("receipt:aseh-055", tree, plan.digest, plan.patch_digest, "pytest -q", passed, issued_at, independent)


def test_closed_round_trip_binds_semantic_plan_digest_and_scope() -> None:
    plan = _plan()
    assert PatchPlan.from_dict(plan.to_dict()) == plan
    tampered = copy.deepcopy(plan.to_dict())
    tampered["semantic_intent"] = "different work"
    with pytest.raises(PatchPlanValidationError, match="digest"):
        PatchPlan.from_dict(tampered)


def test_admits_exact_nonempty_patch_with_current_successful_receipt() -> None:
    plan = _plan()
    result = PatchAdmission().admit(plan, PATCH, _receipt(plan), current_tree=TREE, now_epoch_seconds=NOW)
    assert result.accepted
    assert result.paths == ("pkg/example.py",)
    assert result.plan_digest == plan.digest


@pytest.mark.parametrize(
    ("patch", "receipt_args", "kwargs", "reason"),
    [
        ("", {}, {}, "empty_patch"),
        (PATCH.replace("pkg/example.py", "other/escape.py"), {}, {}, "patch_digest_mismatch"),
        (PATCH.replace("+new = 2", '+password = "super-secret-value"'), {}, {}, "secret_detected"),
        (PATCH.replace("pkg/example.py", "build/example.py"), {}, {}, "generated_artifact"),
        (PATCH, {"passed": False}, {}, "validation_failed"),
        (PATCH, {"tree": "tree:old"}, {}, "stale_receipt"),
        (PATCH, {}, {"conflicting_paths": ("pkg/example.py",)}, "conflict"),
        (PATCH, {}, {"current_tree": "tree:new"}, "stale_plan"),
        (PATCH, {"independent": False}, {}, "self_validation"),
    ],
)
def test_rejects_every_unsafe_or_stale_admission_case(patch, receipt_args, kwargs, reason) -> None:
    plan = _plan()
    current_tree = kwargs.pop("current_tree", TREE)
    result = PatchAdmission().admit(plan, patch, _receipt(plan, **receipt_args), current_tree=current_tree, now_epoch_seconds=NOW, **kwargs)
    assert not result.accepted
    assert reason in result.reason_codes


def test_rejects_expired_receipt_and_receipt_for_a_different_patch() -> None:
    plan = _plan()
    expired = PatchAdmission(maximum_receipt_age_seconds=10).admit(plan, PATCH, _receipt(plan, issued_at=NOW - 11), current_tree=TREE, now_epoch_seconds=NOW)
    assert "stale_receipt" in expired.reason_codes
    changed = PATCH.replace("+new = 2", "+new = 3")
    mismatch = PatchAdmission().admit(plan, changed, _receipt(plan), current_tree=TREE, now_epoch_seconds=NOW)
    assert "patch_digest_mismatch" in mismatch.reason_codes
    assert "receipt_binding_mismatch" in mismatch.reason_codes


def test_malformed_plan_and_path_escape_are_rejected_without_raising() -> None:
    plan = _plan()
    malformed = plan.to_dict()
    malformed["files"] = 1
    result = PatchAdmission().admit(malformed, PATCH, _receipt(plan), current_tree=TREE, now_epoch_seconds=NOW)
    assert result.reason_code == "invalid_plan"
    escaped = PATCH.replace("pkg/example.py", "../../outside.py")
    result = PatchAdmission().admit(plan, escaped, _receipt(plan), current_tree=TREE, now_epoch_seconds=NOW)
    assert not result.accepted
    assert "unsafe_patch" in result.reason_codes
