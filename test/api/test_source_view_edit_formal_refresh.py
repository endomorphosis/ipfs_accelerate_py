"""A mapped native patch retires prior source-bound formal context."""

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime.worker_candidate_context import WorkerCandidateContext
from test.api.test_source_view_edit_candidate import case, invoke  # noqa: F401


def test_native_candidate_formulas_rebuild_after_mapped_patch_and_reject_old_cursor(case):
    context = WorkerCandidateContext(case["views"], ("pkg/app.py", "pkg/other.py"))
    before = context.page(artifact="formal", max_bytes=4096,
        expected_workspace_generation=case["args"]["expected_workspace_generation"])
    assert before["next_cursor"] is not None
    applied = invoke(case)
    generation = applied["workspace_generation"]
    after = context.page(artifact="formal", max_bytes=4096,
        expected_workspace_generation=generation)
    assert after["workspace_generation"] == generation
    assert after["artifact_sha256"] != before["artifact_sha256"]
    assert after["proof_authority"] is False
    assert after["baseline_evidence_applicable"] is False
    assert after["source_obligations_discharged"] == 0
    with pytest.raises(ValueError):
        context.page(artifact="formal", max_bytes=4096,
            expected_workspace_generation=generation, cursor=before["next_cursor"])
    # A refused old formal cursor does not masquerade as a current proof.
    rebuilt = context.page(artifact="formal", max_bytes=4096,
        expected_workspace_generation=generation)
    assert rebuilt["artifact_sha256"] == after["artifact_sha256"]
