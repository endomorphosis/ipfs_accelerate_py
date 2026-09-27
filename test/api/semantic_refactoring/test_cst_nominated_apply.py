from __future__ import annotations

from types import SimpleNamespace

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.autonomy.completion_blocks import (
    clear_completion_blocks,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cst_nominated_apply import (
    apply_nominated_cst_and_stage,
    nominations_from_facade_edits,
    rewrite_nominated_modules,
)


def _move(kind: str) -> SimpleNamespace:
    return SimpleNamespace(
        source_module="old.mod",
        destination_module="new.mod",
        rewrite_kind=kind,
        adapter_kind=kind,
    )


def test_state_init_and_binding_module_moves_preserve_comments() -> None:
    source = "# keep\nold.mod.setup()\nfrom old.mod import boot\n"
    for nomination in (
        _move("state"),
        _move("initialization_order"),
        _move("module_name"),
    ):
        updated, changed = rewrite_nominated_modules(source, (nomination,))
        assert changed is True
        assert updated.startswith("# keep\n")
        assert "new.mod.setup()" in updated
        assert "from new.mod import boot" in updated
        assert "old.mod" not in updated


def test_facade_module_move_preserves_comment_and_does_not_rewrite_cids() -> None:
    edit = SimpleNamespace(kind="facade", source_id="old.mod", destination_id="new.mod")
    cid_edit = SimpleNamespace(
        kind="facade",
        source_id="sha256:" + "a" * 64,
        destination_id="sha256:" + "b" * 64,
    )
    packet = SimpleNamespace(edits=(edit, cid_edit))
    nominations = nominations_from_facade_edits(packet)
    assert len(nominations) == 1
    updated, changed = rewrite_nominated_modules(
        "# keep\nfrom old.mod import Thing\n",
        nominations,
    )
    assert changed is True
    assert updated.startswith("# keep\n")
    assert "from new.mod import Thing" in updated


def test_nominated_module_move_is_staged_without_repo_write_or_cas() -> None:
    clear_completion_blocks()
    packet = SimpleNamespace(tree_id="fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051")
    handoff = apply_nominated_cst_and_stage(
        packet,
        {"pkg/use.py": "# keep\nold.mod.setup()\n"},
        nominations=(_move("pickle"),),
        pre_world_root_cid=cid_for_dag_json({"root": "pre"}),
    )
    assert handoff.applied is True
    assert handoff.blocked is False
    assert handoff.world_root_status == "nominated_persist"
    assert handoff.writes_repository is False
    assert handoff.cas_completed is False
    assert handoff.changes_current_root is False
    assert handoff.completion_authority is False
    assert handoff.libcst_usable is False
