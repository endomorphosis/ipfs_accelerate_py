from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.code_proof_obligations import (
    CandidateDiffEntry,
    CodeProofScopeSet,
    DiffChangeKind,
    ProofScopeKind,
    collect_git_candidate_diff,
    compile_candidate_diff,
    compile_candidate_proof_scopes,
)
from ipfs_accelerate_py.agent_supervisor.conflict_graph import (
    build_conflict_surface,
    build_python_ast_blob_record,
)


PYTHON_SOURCE = """\
from typing import Protocol
import json as json_module

class Store(Protocol):
    def save(self, value: str) -> None: ...

class Worker:
    def __init__(self, store: Store):
        self.store = store
        self.state = "idle"

    def run(self, value: str) -> None:
        self.state = "running"
        self.store.save(json_module.dumps(value))
"""


def _values(compilation, kind: ProofScopeKind) -> set[str]:
    return {scope.value for scope in compilation.by_kind(kind)}


def test_python_diff_compiles_every_typed_ast_scope_and_exact_source_binding() -> None:
    compilation = compile_candidate_proof_scopes(
        [
            CandidateDiffEntry(
                new_path="src/runtime.py",
                change_kind=DiffChangeKind.ADD,
                after_source=PYTHON_SOURCE,
                after_blob_id="git:runtime",
            )
        ]
    )

    symbols = _values(compilation, ProofScopeKind.QUALIFIED_SYMBOL)
    assert {
        "src.runtime.Store",
        "src.runtime.Store.save",
        "src.runtime.Worker",
        "src.runtime.Worker.__init__",
        "src.runtime.Worker.run",
    }.issubset(symbols)
    assert "from typing import Protocol" in _values(compilation, ProofScopeKind.IMPORT)
    assert "json_module.dumps" in _values(compilation, ProofScopeKind.CALL)
    assert "self.store.save" in _values(compilation, ProofScopeKind.CALL)
    assert any(
        value.startswith("self.state:assign:")
        for value in _values(compilation, ProofScopeKind.STATE_TRANSITION)
    )
    assert any(
        value.startswith("src.runtime.Store(Protocol)")
        for value in _values(compilation, ProofScopeKind.INTERFACE)
    )
    assert compilation.changed_paths == ("src/runtime.py",)
    assert compilation.source_hashes[0].startswith("sha256:")
    assert all(
        scope.after_source_hash == compilation.source_hashes[0]
        for scope in compilation.scopes
    )
    assert len(compilation.by_kind(ProofScopeKind.CHANGED_PATH)) == 1
    assert compilation.conservative is False
    assert json.loads(compilation.to_json())["scope_set_id"] == compilation.scope_set_id
    assert CodeProofScopeSet.from_json(compilation.to_json()).scope_ids == compilation.scope_ids


def test_modified_file_selects_changed_symbols_and_their_calls_and_transitions() -> None:
    before = """\
def stable():
    return 1

class Worker:
    def run(self):
        self.state = "idle"
        return helper()
"""
    after = before.replace('"idle"', '"running"')
    compilation = compile_candidate_proof_scopes(
        [
            {
                "path": "pkg/worker.py",
                "status": "modified",
                "before_source": before,
                "after_source": after,
            }
        ]
    )

    symbols = _values(compilation, ProofScopeKind.QUALIFIED_SYMBOL)
    assert "pkg.worker.Worker.run" in symbols
    assert "pkg.worker.stable" not in symbols
    assert "helper" in _values(compilation, ProofScopeKind.CALL)
    state_scopes = compilation.by_kind(ProofScopeKind.STATE_TRANSITION)
    assert any(scope.owner_symbol == "pkg.worker.Worker.run" for scope in state_scopes)


def test_modified_file_preserves_removed_facts_and_both_sides_of_changed_symbols() -> None:
    before = """\
import json

def removed(value: str) -> str:
    return json.dumps(value)

class Worker:
    def run(self):
        self.state = "idle"
        return old_helper()
"""
    after = """\
class Worker:
    def run(self):
        self.state = "running"
        return new_helper()
"""
    compilation = compile_candidate_proof_scopes(
        [
            {
                "path": "pkg/worker.py",
                "status": "modified",
                "before_source": before,
                "after_source": after,
            }
        ]
    )

    removed_symbols = {
        scope.value: scope.delta
        for scope in compilation.by_kind(ProofScopeKind.QUALIFIED_SYMBOL)
        if scope.delta == "removed"
    }
    assert removed_symbols["pkg.worker.removed"] == "removed"
    assert any(
        scope.value == "import json" and scope.delta == "removed"
        for scope in compilation.by_kind(ProofScopeKind.IMPORT)
    )
    assert any(
        scope.value == "old_helper" and scope.delta == "removed"
        for scope in compilation.by_kind(ProofScopeKind.CALL)
    )
    run_versions = {
        scope.delta
        for scope in compilation.by_kind(ProofScopeKind.QUALIFIED_SYMBOL)
        if scope.value == "pkg.worker.Worker.run"
    }
    assert run_versions == {"modified_before", "modified_after"}
    assert {
        scope.delta
        for scope in compilation.by_kind(ProofScopeKind.STATE_TRANSITION)
        if scope.owner_symbol == "pkg.worker.Worker.run"
    } == {"removed", "added"}


def test_cold_and_warm_blob_scans_have_identical_canonical_scope_identities() -> None:
    entries = [
        {
            "new_path": "src/service.py",
            "status": "add",
            "after_source": PYTHON_SOURCE,
            "after_blob_id": "blob:one",
        }
    ]
    cold = compile_candidate_proof_scopes(entries)
    warm = compile_candidate_proof_scopes(entries, ast_records=cold.ast_records)
    surface = build_conflict_surface(
        {
            "task_id": "REF-248",
            "outputs": ["src/service.py"],
            "ast_records": [record.to_dict() for record in cold.ast_records],
        }
    )
    from_surface = compile_candidate_proof_scopes(entries, conflict_surfaces=[surface])

    assert cold.scope_ids == warm.scope_ids == from_surface.scope_ids
    assert cold.scope_set_id == warm.scope_set_id == from_surface.scope_set_id
    assert cold.stats.parsed_blob_count == 1
    assert warm.stats.parsed_blob_count == 0
    assert warm.stats.reused_blob_count == 1
    assert surface.blob_identities == ["blob:one"]
    assert surface.ast_records[0]["record_id"] == cold.ast_records[0].record_id


def test_conflict_surface_reuses_objective_records_by_blob_without_leaking_snapshot(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "src" / "service.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_text(PYTHON_SOURCE, encoding="utf-8")
    unrelated = build_python_ast_blob_record(
        "def unrelated():\n    return True\n",
        blob_identity="blob:unrelated",
    ).to_dict()
    unrelated["root_relative_path"] = "src/unrelated.py"
    objective_record = {
        "record_schema_version": 2,
        "root_relative_path": "src/service.py",
        "suffix": ".py",
        "blob_hash": "blob:objective-service",
        "source_sha1": "legacy-sha1",
        "evidence_text": PYTHON_SOURCE,
        "symbols_json": "[]",
        "ast_kind": "python_ast",
    }

    surface = build_conflict_surface(
        {
            "task_id": "REF-248",
            "outputs": ["src/service.py"],
            "ast_records": [objective_record, unrelated],
        },
        repo_root=tmp_path,
    )
    entries = [
        {
            "new_path": "src/service.py",
            "status": "add",
            "after_source": PYTHON_SOURCE,
            "after_blob_id": "blob:objective-service",
        }
    ]
    cold = compile_candidate_proof_scopes(entries)
    warm = compile_candidate_proof_scopes(entries, conflict_surfaces=[surface])

    assert surface.blob_identities == ["blob:objective-service"]
    assert "unrelated" not in surface.ast_symbols
    assert cold.scope_ids == warm.scope_ids
    assert warm.stats.reused_blob_count == 1
    assert warm.stats.parsed_blob_count == 0


def test_rename_reuses_one_blob_but_requalifies_old_and_new_modules() -> None:
    cold = compile_candidate_proof_scopes(
        [
            {
                "old_path": "old/service.py",
                "new_path": "new/service.py",
                "status": "R100",
                "before_source": PYTHON_SOURCE,
                "after_source": PYTHON_SOURCE,
                "before_blob_id": "same-git-blob",
                "after_blob_id": "same-git-blob",
            }
        ]
    )
    symbols = _values(cold, ProofScopeKind.QUALIFIED_SYMBOL)

    assert "old.service.Worker.run" in symbols
    assert "new.service.Worker.run" in symbols
    assert cold.stats.parsed_blob_count == 1
    assert cold.stats.reused_blob_count == 1
    assert cold.changed_paths == ("new/service.py", "old/service.py")
    assert "rename_requires_reference_validation" in cold.conservative_reasons
    assert {scope.delta for scope in cold.by_kind(ProofScopeKind.QUALIFIED_SYMBOL)} == {
        "added",
        "removed",
    }


def test_deletes_generated_syntax_failures_and_non_python_changes_fail_closed() -> None:
    compilation = compile_candidate_proof_scopes(
        [
            {
                "old_path": "src/deleted.py",
                "status": "delete",
                "before_source": "def removed():\n    return True\n",
            },
            {
                "path": "generated/client.py",
                "status": "modify",
                "before_source": "VALUE = 1\n",
                "after_source": "VALUE = 2\n",
            },
            {
                "path": "src/broken.py",
                "status": "modify",
                "before_source": "def valid():\n    return 1\n",
                "after_source": "def broken(:\n",
            },
            {
                "path": "schema/api.json",
                "status": "modify",
                "before_source": '{"v": 1}',
                "after_source": '{"v": 2}',
            },
        ]
    )

    assert {
        "deleted_path",
        "generated_file",
        "non_python_change",
    }.issubset(compilation.conservative_reasons)
    assert any(reason.startswith("after_syntax_error:") for reason in compilation.conservative_reasons)
    conservative_paths = {
        scope.path for scope in compilation.by_kind(ProofScopeKind.CONSERVATIVE_FILE)
    }
    assert conservative_paths == {
        "generated/client.py",
        "schema/api.json",
        "src/broken.py",
    }
    deleted_symbols = {
        scope.value
        for scope in compilation.by_kind(ProofScopeKind.QUALIFIED_SYMBOL)
        if scope.path == "src/deleted.py"
    }
    assert deleted_symbols == {"src.deleted.removed"}
    assert all(
        scope.conservative
        for scope in compilation.scopes
        if scope.path == "src/deleted.py"
    )


def test_unified_diff_without_full_sources_is_explicitly_conservative() -> None:
    compilation = compile_candidate_diff(
        """\
diff --git a/src/value.py b/src/value.py
index 123..456 100644
--- a/src/value.py
+++ b/src/value.py
@@ -1 +1 @@
-VALUE = 1
+VALUE = 2
"""
    )
    assert compilation.changed_paths == ("src/value.py",)
    assert compilation.conservative_reasons == ("missing_source",)
    assert compilation.by_kind(ProofScopeKind.CONSERVATIVE_FILE)


def test_git_candidate_collection_covers_modify_rename_delete_and_untracked(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.email", "test@example.invalid"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "keep.py").write_text("def keep():\n    return 1\n", encoding="utf-8")
    (repo / "rename.py").write_text("def moved():\n    return 1\n", encoding="utf-8")
    (repo / "delete.py").write_text("def gone():\n    return 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "base"], cwd=repo, check=True)

    (repo / "keep.py").write_text("def keep():\n    return 2\n", encoding="utf-8")
    (repo / "rename.py").rename(repo / "moved.py")
    (repo / "delete.py").unlink()
    (repo / "new.py").write_text("def new():\n    return 1\n", encoding="utf-8")

    entries = collect_git_candidate_diff(repo)
    compilation = compile_candidate_diff(repo)
    kinds = {entry.change_kind for entry in entries}

    assert {
        DiffChangeKind.ADD,
        DiffChangeKind.MODIFY,
        DiffChangeKind.DELETE,
        DiffChangeKind.RENAME,
    }.issubset(kinds)
    assert compilation.changed_paths == (
        "delete.py",
        "keep.py",
        "moved.py",
        "new.py",
        "rename.py",
    )
    assert compilation.by_kind(ProofScopeKind.QUALIFIED_SYMBOL)
