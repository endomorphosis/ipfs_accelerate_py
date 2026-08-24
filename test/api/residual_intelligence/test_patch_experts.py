from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.local_experts import (
    IndependentValidationReceipt,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.patch_experts import (
    REASON_ISOLATED_WORKTREE_REQUIRED,
    REASON_MAX_LINES,
    REASON_PATH_OUT_OF_SCOPE,
    REASON_TEST_DELETION,
    PatchExpertAdapter,
    PatchScopePolicy,
    PatchSketchIR,
    TestSketchIR,
)


def policy() -> PatchScopePolicy:
    return PatchScopePolicy(
        allowed_paths=("ipfs_accelerate_py/module.py", "test/api/test_module.py"),
        maximum_changed_lines=40,
    )


def sketch(**overrides: object) -> PatchSketchIR:
    payload = {
        "base_tree_cid": "tree:abc",
        "paths": ("ipfs_accelerate_py/module.py",),
        "symbol_ids": ("module.fn",),
        "operations": ("replace_function",),
        "changed_lines": 12,
    }
    payload.update(overrides)
    return PatchSketchIR(**payload)


def tests() -> TestSketchIR:
    return TestSketchIR(validation_ids=("pytest:focused",), added_tests=("test_fn",))


def test_scope_and_line_bounds() -> None:
    adapter = PatchExpertAdapter(policy=policy(), isolated_worktree=True)
    result = adapter.nominate(
        sketch(),
        tests(),
        validation=IndependentValidationReceipt(
            validator_identity="validator:patch@1", accepted=True
        ),
    )
    assert result["candidate_only"] is True
    assert result["rollback"]["mode"] == "discard_fenced_worktree"
    with pytest.raises(ResidualIntelligenceError, match=REASON_PATH_OUT_OF_SCOPE):
        adapter.nominate(sketch(paths=("docs/secrets/key.pem",)), tests())


def test_prohibits_test_deletion_binary_and_unowned_worktree() -> None:
    with pytest.raises(ResidualIntelligenceError, match=REASON_TEST_DELETION):
        TestSketchIR(validation_ids=("pytest",), deleted_tests=("test_old",))
    with pytest.raises(ResidualIntelligenceError, match=REASON_PATH_OUT_OF_SCOPE):
        sketch(paths=("pkg/native.so",))
    with pytest.raises(ResidualIntelligenceError, match=REASON_MAX_LINES):
        PatchExpertAdapter(policy=policy(), isolated_worktree=True).nominate(
            sketch(changed_lines=400),
            tests(),
        )
    with pytest.raises(ResidualIntelligenceError, match=REASON_ISOLATED_WORKTREE_REQUIRED):
        PatchExpertAdapter(policy=policy(), isolated_worktree=False).nominate(
            sketch(), tests()
        )
    with pytest.raises(ResidualIntelligenceError, match="shell"):
        sketch(operations=("shell",))
