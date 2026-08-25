"""EAAEF-104: terminate only when roots, tests, proofs, queue and claims agree."""

from __future__ import annotations

from ipfs_accelerate_py.agent_supervisor.runtime.external_fixed_point import terminate


def test_complete_only_when_all_gates_hold() -> None:
    done = terminate(
        goals_complete=True,
        tests_current=True,
        proofs_current=True,
        invalidations_empty=True,
        merge_queue_empty=True,
        claims_empty=True,
        source_root="sha256:" + "a" * 64,
        semantic_root="sha256:" + "b" * 64,
    )
    assert done["terminal"] == "completed"
    open_queue = terminate(
        goals_complete=True,
        tests_current=True,
        proofs_current=True,
        invalidations_empty=True,
        merge_queue_empty=False,
        claims_empty=True,
        source_root="sha256:" + "a" * 64,
        semantic_root="sha256:" + "b" * 64,
    )
    assert open_queue["terminal"] == "not_complete"
