"""ASEH-055: PatchPlan and merge-admission safety contract."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import patch_admission
from ipfs_accelerate_py.agent_supervisor.merge.patch_admission import (
    PatchAdmissionReason,
    ValidationReceipt,
    admit_patch_plan,
)
from ipfs_accelerate_py.agent_supervisor.merge.patch_plan import (
    PatchFile,
    PatchPlan,
    PatchPlanValidationError,
)


PATCH = """diff --git a/pkg/widget.py b/pkg/widget.py
index 1111111..2222222 100644
--- a/pkg/widget.py
+++ b/pkg/widget.py
@@ -1 +1 @@
-def old(): return 1
+def current(): return 2
"""


def _plan(patch: str = PATCH) -> PatchPlan:
    return PatchPlan.create(
        base_tree_id="tree:base", semantic_intent="replace the legacy widget entrypoint",
        files=(PatchFile("pkg/widget.py", ("current",)),),
        preconditions=("tree is base",), postconditions=("entrypoint is current",),
        invariants=("public contract remains available",), tests_proofs=("pytest widget",),
        scope=("pkg/",), patch=patch, proposer_id="candidate:worker-1",
    )


def _receipt(plan: PatchPlan, **changes: object) -> ValidationReceipt:
    data: dict[str, object] = {
        "base_tree_id": plan.base_tree_id, "candidate_tree_id": "tree:candidate",
        "plan_digest": plan.plan_digest, "patch_digest": plan.patch_digest, "passed": True,
        "validator_id": "validator:independent", "validation_command": "pytest -q pkg",
        "receipt_id": "receipt:current",
    }
    data.update(changes)
    return ValidationReceipt(**data)  # type: ignore[arg-type]


def _admit(plan: PatchPlan, patch: str = PATCH, **kwargs: object):
    return admit_patch_plan(
        plan, patch, current_base_tree_id=kwargs.pop("current_base_tree_id", "tree:base"),
        current_candidate_tree_id=kwargs.pop("current_candidate_tree_id", "tree:candidate"),
        validation_receipt=kwargs.pop("validation_receipt", _receipt(plan)), **kwargs,
    )


def test_closed_plan_round_trip_canonicalizes_and_detects_tampering() -> None:
    plan = _plan()
    loaded = PatchPlan.from_dict(plan.to_dict())
    assert loaded == plan
    assert loaded.plan_digest == plan.plan_digest
    schema_path = Path(__file__).parents[4] / "ipfs_accelerate_py/agent_supervisor/merge/schemas/patch_plan.schema.json"
    assert json.loads(schema_path.read_text(encoding="utf-8"))["$id"] == "aseh/patch-plan@1"
    forged = copy.deepcopy(plan.to_dict())
    forged["semantic_intent"] = "different intent"
    with pytest.raises(PatchPlanValidationError, match="digest mismatch"):
        PatchPlan.from_dict(forged)


def test_digest_tree_scope_and_nonempty_patch_are_required() -> None:
    plan = _plan()
    assert _admit(plan).admitted
    assert PatchAdmissionReason.EMPTY_PATCH in _admit(_plan(""), "").reasons
    assert PatchAdmissionReason.PATCH_DIGEST_MISMATCH in _admit(plan, PATCH + "# altered\n").reasons
    assert PatchAdmissionReason.STALE_PLAN in _admit(plan, current_base_tree_id="tree:new").reasons
    escaped = PATCH.replace("pkg/widget.py", "other/widget.py")
    result = _admit(_plan(escaped), escaped)
    assert PatchAdmissionReason.SCOPE_ESCAPE in result.reasons
    assert PatchAdmissionReason.FILE_SET_MISMATCH in result.reasons


@pytest.mark.parametrize(
    ("path", "content", "reason"),
    (
        ("pkg/__pycache__/widget.pyc", "binary content", PatchAdmissionReason.GENERATED_ARTIFACT),
        ("pkg/widget.py", 'api_key = "abcdefghijklmnopqrstuvwx"', PatchAdmissionReason.SECRET_DETECTED),
    ),
)
def test_secret_and_generated_artifact_checks_fail_closed(path: str, content: str, reason: PatchAdmissionReason) -> None:
    patch = PATCH.replace("pkg/widget.py", path).replace("def current(): return 2", content)
    plan = PatchPlan.create(
        base_tree_id="tree:base", semantic_intent="test safety", files=(PatchFile(path, ("artifact",)),),
        preconditions=("pre",), postconditions=("post",), invariants=("invariant",),
        tests_proofs=("test",), scope=("pkg/",), patch=patch, proposer_id="candidate:worker-1",
    )
    assert reason in _admit(plan, patch).reasons


def test_current_receipt_failed_validation_self_validation_and_conflicts_deny() -> None:
    plan = _plan()
    assert PatchAdmissionReason.MISSING_VALIDATION_RECEIPT in _admit(plan, validation_receipt=None).reasons
    assert PatchAdmissionReason.FAILED_VALIDATION in _admit(plan, validation_receipt=_receipt(plan, passed=False)).reasons
    assert PatchAdmissionReason.STALE_VALIDATION_RECEIPT in _admit(plan, validation_receipt=_receipt(plan, patch_digest="sha256:" + "0" * 64)).reasons
    assert PatchAdmissionReason.SELF_VALIDATION in _admit(plan, validation_receipt=_receipt(plan, validator_id=plan.proposer_id)).reasons
    assert PatchAdmissionReason.CONFLICT in _admit(plan, conflicting_paths=("pkg/widget.py",)).reasons
    assert PatchAdmissionReason.STALE_VALIDATION_RECEIPT in _admit(plan, current_candidate_tree_id="tree:other").reasons
    assert PatchAdmissionReason.MISSING_CURRENT_CANDIDATE_TREE in _admit(plan, current_candidate_tree_id=None).reasons


def test_admission_budgets_are_independent_fail_closed_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    plan = _plan()
    monkeypatch.setattr(patch_admission, "MAX_PATCH_BYTES", len(PATCH) - 1)
    assert PatchAdmissionReason.PATCH_BUDGET_EXCEEDED in _admit(plan).reasons
    monkeypatch.setattr(patch_admission, "MAX_PATCH_BYTES", len(PATCH) + 1)
    monkeypatch.setattr(patch_admission, "MAX_SINGLE_FILE_BYTES", len(PATCH) - 1)
    assert PatchAdmissionReason.SINGLE_FILE_BUDGET_EXCEEDED in _admit(plan).reasons
