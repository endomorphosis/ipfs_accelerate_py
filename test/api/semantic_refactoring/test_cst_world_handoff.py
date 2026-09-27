from __future__ import annotations

from types import SimpleNamespace

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.autonomy.completion_blocks import (
    clear_completion_blocks,
    publish_completion_blocks,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cst_world_handoff import (
    clear_cst_handoffs,
    last_cst_handoffs,
    stage_cst_world_root,
)


def _result() -> SimpleNamespace:
    return SimpleNamespace(
        tree_id="fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051",
        result_cid=cid_for_dag_json({"artifact": "cst-extraction"}),
        mutated=False,
        writes_repository=False,
        libcst_usable=False,
        can_authorize_completion=False,
        source_maps=(SimpleNamespace(origin_path="a.py"),),
        sources={"a.py": "def moved():\n    return 1\n"},
    )


def test_cst_result_is_staged_without_cas_or_repo_write() -> None:
    clear_completion_blocks()
    clear_cst_handoffs()
    handoff = stage_cst_world_root(_result(), expected_generation=1)
    recorded = last_cst_handoffs()
    assert recorded[-1] is handoff
    assert recorded[-1].sources == (
        ("a.py", "def moved():\n    return 1\n"),
    )
    assert recorded[-1].to_dict()["source_paths"] == ["a.py"]
    assert recorded[-1].cas_completed is False
    assert handoff.applied is True
    assert handoff.blocked is False
    assert handoff.writes_repository is False
    assert handoff.cas_completed is False
    assert handoff.changes_current_root is False
    assert handoff.completion_authority is False
    assert handoff.libcst_usable is False
    assert handoff.outbox_nomination_cid
    assert handoff.semantic_world_root_cid
    record = handoff.to_dict()
    assert record["accepted_as_authority"] is False


def test_outstanding_completion_block_refuses_cst_world_handoff() -> None:
    publish_completion_blocks(undeclared_cst_transform=True)
    handoff = stage_cst_world_root(_result())
    assert handoff.applied is False
    assert handoff.blocked is True
    assert handoff.reason == "undeclared_cst_transform"
    assert handoff.completion_authority is False
    clear_completion_blocks()


def test_kit_world_root_is_nominated_without_cas_of_current_root() -> None:
    clear_completion_blocks()
    pre = cid_for_dag_json({"root": "current"})
    handoff = stage_cst_world_root(
        _result(),
        expected_generation=2,
        pre_world_root_cid=pre,
    )
    assert handoff.applied is True
    assert handoff.world_root_status == "nominated_persist"
    assert handoff.cas_completed is False
    assert handoff.changes_current_root is False
    assert handoff.writes_repository is False


def test_bound_kit_store_commits_root_and_publishes_vfs(tmp_path) -> None:
    from ipfs_kit_py.mcp_server.mcplusplus.coordination_storage import (
        DurableCoordinationStore,
    )

    clear_completion_blocks()
    clear_cst_handoffs()
    store_dir = tmp_path / "kit-cst"
    coordination = DurableCoordinationStore(store_dir)
    try:
        from ipfs_kit_py.semantic_world_store import SemanticWorldStore

        facade = SemanticWorldStore(coordination)
        handoff = stage_cst_world_root(_result(), kit_store=facade)
        assert handoff.applied is True
        assert handoff.blocked is False
        assert handoff.reason == "kit_root_committed"
        assert handoff.vfs_published is True
        assert handoff.kit_generation == 1
        assert handoff.kit_root_cid
        assert handoff.cas_completed is False
        assert handoff.writes_repository is False
        assert handoff.changes_current_root is False
        assert handoff.completion_authority is False
        record = handoff.to_dict()
        assert record["supervisor_accepted"] is False
        assert record["vfs_published"] is True
        current = facade.current_world_root()
        assert current.root_cid == handoff.kit_root_cid
        assert current.supervisor_accepted is False
        assert facade.outbox.read_published() == handoff.kit_root_cid.encode("utf-8")
        second = stage_cst_world_root(
            _result(),
            kit_store=facade,
            kit_predecessor_cid="bafyrei-not-the-current-root",
        )
        assert second.applied is False
        assert second.reason == "world_root_predecessor_mismatch"
        assert facade.current_world_root().generation == 1
        assert facade.outbox.read_published() == handoff.kit_root_cid.encode("utf-8")
        root_cid = handoff.kit_root_cid
    finally:
        coordination.close()
    reopened = DurableCoordinationStore(store_dir)
    try:
        restored_store = SemanticWorldStore(reopened)
        restored = restored_store.current_world_root()
        assert restored.root_cid == root_cid
        assert restored.generation == 1
        assert restored.supervisor_accepted is False
        assert restored_store.outbox.read_published() == root_cid.encode("utf-8")
        from ipfs_kit_py.core.vfs.contracts import VFSOperationKind
        from ipfs_kit_py.core.vfs.service import make_op

        source = restored_store.outbox.vfs.execute(
            make_op(
                VFSOperationKind.READ,
                operation_id="read-reopened-source",
                path="sources/a.py",
            )
        )
        assert source.success is True
        assert source.data == b"def moved():\n    return 1\n"
        from ipfs_datasets_py.logic.software_contracts.content import cid_for_bytes

        assert reopened.get_bytes(cid_for_bytes(source.data)) == source.data
    finally:
        reopened.close()


def test_missing_source_map_is_not_a_cst_apply() -> None:
    clear_completion_blocks()
    result = _result()
    result.source_maps = ()
    handoff = stage_cst_world_root(result)
    assert handoff.blocked is True
    assert handoff.reason == "source_map_not_preserved"
    assert handoff.applied is False
