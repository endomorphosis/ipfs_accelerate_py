from __future__ import annotations

from types import SimpleNamespace

from ipfs_accelerate_py.utils.cid_utils import cid_for_dag_json
from ipfs_accelerate_py.agent_supervisor.autonomy.completion_blocks import (
    clear_completion_blocks,
)
from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cst_import_apply import (
    apply_import_cst_and_stage,
    apply_reexport_cst_and_stage,
    insert_reexports_with_cst,
    rewrite_imports_with_cst,
)


def test_cst_callsite_rewrite_preserves_leading_comment() -> None:
    rewrite = SimpleNamespace(
        rewrite_kind="callsite",
        source_module="old.mod",
        destination_module="new.mod",
        symbol_id="Thing",
    )
    source = "# keep\nold.mod.Thing()\nother.Thing()\n"
    updated, changed = rewrite_imports_with_cst(source, (rewrite,))
    assert changed is True
    assert updated.startswith("# keep\n")
    assert "new.mod.Thing()" in updated
    assert "other.Thing()" in updated


def test_cst_import_rewrite_preserves_leading_comment() -> None:
    rewrite = SimpleNamespace(
        rewrite_kind="import",
        source_module="old.mod",
        destination_module="new.mod",
        symbol_id="Thing",
    )
    source = "# keep\nfrom old.mod import Thing\n\nThing()\n"
    updated, changed = rewrite_imports_with_cst(source, (rewrite,))
    assert changed is True
    assert updated.startswith("# keep\n")
    assert "from new.mod import Thing" in updated
    assert "Thing()" in updated


def test_reexport_is_inserted_after_comment_and_staged() -> None:
    clear_completion_blocks()
    plan = SimpleNamespace(
        source_module="old.mod",
        destination_module="new.mod",
        symbol_ids=("Thing",),
        write_paths=("pkg/mod.py",),
    )
    updated, changed = insert_reexports_with_cst(
        "# keep\nclass Facade:\n    pass\n",
        (plan,),
    )
    assert changed is True
    assert updated.startswith("# keep\n")
    assert "from new.mod import Thing\n" in updated
    assert "class Facade" in updated
    packet = SimpleNamespace(tree_id="fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051")
    handoff = apply_reexport_cst_and_stage(
        packet,
        {"pkg/mod.py": "# keep\nclass Facade:\n    pass\n"},
        plans=(plan,),
        pre_world_root_cid=cid_for_dag_json({"root": "pre"}),
    )
    assert handoff.applied is True
    assert handoff.world_root_status == "nominated_persist"
    assert handoff.writes_repository is False
    assert handoff.cas_completed is False
    assert handoff.changes_current_root is False
    assert handoff.completion_authority is False


def test_applied_import_is_staged_to_world_root_without_repo_write() -> None:
    clear_completion_blocks()
    rewrite = SimpleNamespace(
        rewrite_kind="import",
        source_module="old.mod",
        destination_module="new.mod",
        symbol_id="Thing",
    )
    packet = SimpleNamespace(tree_id="fbc6fa1ddefb2f9ecb7b5c718d618e3b60aa3051")
    handoff = apply_import_cst_and_stage(
        packet,
        {"pkg/use.py": "from old.mod import Thing\n"},
        rewrites=(rewrite,),
        pre_world_root_cid=cid_for_dag_json({"root": "pre"}),
    )
    assert handoff.applied is True
    assert handoff.blocked is False
    assert handoff.world_root_status == "nominated_persist"
    assert handoff.writes_repository is False
    assert handoff.cas_completed is False
    assert handoff.completion_authority is False
    assert handoff.libcst_usable is False
