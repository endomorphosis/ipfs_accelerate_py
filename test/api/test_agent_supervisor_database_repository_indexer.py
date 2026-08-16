"""Tests for DatabaseRepositoryIndexer@1 / ASTInvalidation@1 (DQP-021).

Evidence subset: watcher loss/coalescing, rename, delete, untracked policy,
submodule change, partial scan crash, clean rebuild equivalence.

Acceptance:

* Incremental result equals a clean full scan for the same snapshot
* Missed notifications are recovered by reconciliation
* A partial scan never advances the authoritative snapshot head
* Dependent facts cannot remain current after source/parser/policy drift
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.database_repository_indexer import (
    AST_INVALIDATION_INTERFACE,
    DATABASE_REPOSITORY_INDEXER_INTERFACE,
    ASTInvalidation,
    ChangeKind,
    DatabaseRepositoryIndexer,
    FactCurrency,
    FactKind,
    InvalidationReason,
    ScanMode,
    ScanStatus,
    duckdb_available,
    open_database_repository_indexer,
)
from ipfs_accelerate_py.agent_supervisor.analysis.duckdb_ast_index import (
    SourceFileSpec,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for DatabaseRepositoryIndexer hermetic tests",
)


PYTHON_SERVICE_V1 = """\
class Service:
    def dispatch(self, request):
        return request
"""

PYTHON_SERVICE_V2 = """\
class Service:
    def dispatch(self, request):
        self.status = "running"
        return request
"""

PYTHON_CONSUMER = """\
from src.service import Service

def consume(request):
    service = Service()
    return service.dispatch(request)
"""

PYTHON_HELPER = """\
def helper():
    return 42
"""


def _open(tmp_path: Path) -> DatabaseRepositoryIndexer:
    return open_database_repository_indexer(tmp_path / "indexer.duckdb")


def _files(*pairs: tuple[str, str]) -> list[SourceFileSpec]:
    return [SourceFileSpec(path=path, content=body) for path, body in pairs]


def test_interface_identities() -> None:
    assert DATABASE_REPOSITORY_INDEXER_INTERFACE == "DatabaseRepositoryIndexer@1"
    assert AST_INVALIDATION_INTERFACE == "ASTInvalidation@1"
    assert DatabaseRepositoryIndexer.INTERFACE == DATABASE_REPOSITORY_INDEXER_INTERFACE
    inv = ASTInvalidation(
        invalidation_id="",
        worktree_id="worktree:wt-1",
        path="src/a.py",
        reason=InvalidationReason.PATH_DELETED,
        prior_content_digest="sha256:" + "a" * 64,
    )
    assert inv.interface == AST_INVALIDATION_INTERFACE
    assert inv.to_dict()["authority"] == "derived_evidence"


def test_cold_import_and_construction_have_no_side_effects() -> None:
    store = DatabaseRepositoryIndexer("/tmp/should-not-exist-until-open.duckdb")
    assert store.is_open is False


def test_full_scan_advances_authoritative_head(tmp_path: Path) -> None:
    with _open(tmp_path) as indexer:
        result = indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:base",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V1),
                ("src/consumer.py", PYTHON_CONSUMER),
            ),
        )
        assert result.status is ScanStatus.SUCCEEDED
        assert result.mode is ScanMode.FULL
        assert result.head_advanced is True
        assert result.complete is True
        assert result.snapshot is not None
        head = indexer.get_authoritative_head("worktree:wt-1")
        assert head is not None
        assert head.snapshot_id == result.snapshot_id
        assert head.tree_id == "tree:base"
        assert set(head.file_ledger) == {"src/service.py", "src/consumer.py"}
        symbols = indexer.ast_index.list_symbols(result.snapshot_id)
        names = {item.qualified_name for item in symbols}
        assert "Service" in names
        assert "Service.dispatch" in names
        assert "consume" in names
        meta = indexer.metadata()
        assert meta["interface"] == DATABASE_REPOSITORY_INDEXER_INTERFACE
        receipt = indexer.get_coverage_receipt(result.scan_run_id)
        assert receipt is not None
        assert receipt.complete is True
        assert receipt.path_count == 2


def test_incremental_equals_clean_full_scan_for_same_snapshot(
    tmp_path: Path,
) -> None:
    files_v1 = _files(
        ("src/service.py", PYTHON_SERVICE_V1),
        ("src/consumer.py", PYTHON_CONSUMER),
    )
    files_v2 = _files(
        ("src/service.py", PYTHON_SERVICE_V2),
        ("src/consumer.py", PYTHON_CONSUMER),
        ("src/helper.py", PYTHON_HELPER),
    )

    with _open(tmp_path) as indexer:
        base = indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v1",
            files=files_v1,
        )
        assert base.head_advanced is True

        incremental = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v2",
            files=files_v2,
        )
        assert incremental.head_advanced is True
        assert incremental.mode is ScanMode.INCREMENTAL
        assert "src/helper.py" in incremental.delta.added
        assert "src/service.py" in incremental.delta.changed
        assert "src/consumer.py" in incremental.delta.unchanged
        assert incremental.coverage.reused_count >= 1

        # Clean full scan of the same file ledger (distinct tree label so the
        # content-addressed snapshot row is a true independent rebuild) must
        # produce equivalent AST evidence.
        clean = indexer.full_scan(
            worktree_id="worktree:wt-clean",
            repository_id="repo:demo",
            tree_id="tree:v2-clean-rebuild",
            files=files_v2,
        )
        equivalence = indexer.snapshot_equivalence(
            incremental.snapshot_id, clean.snapshot_id
        )
        assert equivalence["equal"] is True
        assert equivalence["symbol_count_left"] == equivalence["symbol_count_right"]
        assert equivalence["file_diff"]["only_left"] == []
        assert equivalence["file_diff"]["only_right"] == []
        assert equivalence["file_diff"]["digest_mismatch"] == []


def test_partial_scan_never_advances_authoritative_head(tmp_path: Path) -> None:
    with _open(tmp_path) as indexer:
        base = indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:base",
            files=_files(("src/service.py", PYTHON_SERVICE_V1)),
        )
        prior_head = indexer.get_authoritative_head("worktree:wt-1")
        assert prior_head is not None
        prior_snapshot = prior_head.snapshot_id

        partial = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:partial",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V2),
                ("src/helper.py", PYTHON_HELPER),
            ),
            crash_after_paths=1,
        )
        assert partial.status is ScanStatus.PARTIAL
        assert partial.head_advanced is False
        assert partial.complete is False
        assert partial.snapshot is None
        assert partial.coverage.complete is False
        assert "partial" in str(partial.coverage.body.get("reason") or "partial")

        head = indexer.get_authoritative_head("worktree:wt-1")
        assert head is not None
        assert head.snapshot_id == prior_snapshot
        assert head.tree_id == "tree:base"
        # Prior snapshot evidence remains the only authoritative view.
        symbols = indexer.ast_index.list_symbols(prior_snapshot)
        assert {item.qualified_name for item in symbols} == {
            "Service",
            "Service.dispatch",
        }
        # Cursor may record the failed attempt without promoting the head.
        cursor = indexer.get_scan_cursor("worktree:wt-1")
        assert cursor is not None
        assert cursor.last_scan_run_id == partial.scan_run_id


def test_missed_notifications_recovered_by_reconciliation(tmp_path: Path) -> None:
    with _open(tmp_path) as indexer:
        indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v1",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V1),
                ("src/consumer.py", PYTHON_CONSUMER),
            ),
        )

        # Watcher loss: source changed with no notification delivered.
        recovered = indexer.reconcile(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v2",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V2),
                ("src/consumer.py", PYTHON_CONSUMER),
            ),
        )
        assert recovered.mode is ScanMode.RECONCILE
        assert recovered.head_advanced is True
        assert "src/service.py" in recovered.delta.changed
        assert recovered.coverage.complete is True
        head = indexer.get_authoritative_head("worktree:wt-1")
        assert head is not None
        assert head.tree_id == "tree:v2"
        invalidations = indexer.list_invalidations(
            "worktree:wt-1", path="src/service.py"
        )
        assert any(
            item.reason is InvalidationReason.SOURCE_CHANGED
            for item in invalidations
        )


def test_notification_coalescing_and_apply(tmp_path: Path) -> None:
    with _open(tmp_path) as indexer:
        indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v1",
            files=_files(("src/service.py", PYTHON_SERVICE_V1)),
        )
        first = indexer.notify_change(
            worktree_id="worktree:wt-1",
            path="src/service.py",
            change_kind=ChangeKind.CHANGED,
        )
        second = indexer.notify_change(
            worktree_id="worktree:wt-1",
            path="src/service.py",
            change_kind=ChangeKind.CHANGED,
        )
        # Second notification coalesces into the open one for the path.
        assert second.coalesced_into == first.notification_id
        open_notes = indexer.list_open_notifications("worktree:wt-1")
        assert len(open_notes) == 1
        assert open_notes[0].notification_id == first.notification_id

        result = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v2",
            files=_files(("src/service.py", PYTHON_SERVICE_V2)),
            apply_notifications=True,
        )
        assert result.head_advanced is True
        assert result.coverage.notification_applied_count >= 1
        assert indexer.list_open_notifications("worktree:wt-1") == ()


def test_rename_and_delete_invalidate_path_bindings(tmp_path: Path) -> None:
    with _open(tmp_path) as indexer:
        base = indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v1",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V1),
                ("src/old_name.py", PYTHON_HELPER),
                ("src/gone.py", "def gone():\n    return 0\n"),
            ),
        )
        assert base.head_advanced is True

        next_scan = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v2",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V1),
                # Same bytes as old_name.py at a new path => rename.
                ("src/new_name.py", PYTHON_HELPER),
            ),
        )
        assert "src/gone.py" in next_scan.delta.deleted
        assert "src/new_name.py" in next_scan.delta.renamed
        assert next_scan.delta.renamed["src/new_name.py"] == "src/old_name.py"
        reasons = {
            (item.path, item.reason)
            for item in next_scan.invalidations
        }
        assert ("src/gone.py", InvalidationReason.PATH_DELETED) in reasons
        assert ("src/old_name.py", InvalidationReason.PATH_RENAMED) in reasons

        # New snapshot has the renamed path and not the deleted one.
        paths = {
            item["path"]
            for item in indexer.ast_index.list_files(next_scan.snapshot_id)
        }
        assert "src/new_name.py" in paths
        assert "src/gone.py" not in paths
        assert "src/old_name.py" not in paths


def test_untracked_and_submodule_policy_notifications(tmp_path: Path) -> None:
    with _open(tmp_path) as indexer:
        indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v1",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V1),
                # Previously indexed path that later becomes a gitlink.
                ("vendor/lib/mod.py", "def vendored():\n    return 1\n"),
            ),
        )
        untracked = indexer.notify_change(
            worktree_id="worktree:wt-1",
            path="scratch/local.py",
            change_kind=ChangeKind.UNTRACKED,
        )
        submodule = indexer.notify_change(
            worktree_id="worktree:wt-1",
            path="vendor/lib/mod.py",
            change_kind=ChangeKind.SUBMODULE,
            content_digest="sha256:" + "b" * 64,
            coalesce=False,
        )
        assert untracked.change_kind is ChangeKind.UNTRACKED
        assert submodule.change_kind is ChangeKind.SUBMODULE
        open_notes = indexer.list_open_notifications("worktree:wt-1")
        kinds = {item.change_kind for item in open_notes}
        assert ChangeKind.UNTRACKED in kinds
        assert ChangeKind.SUBMODULE in kinds

        # Untracked files may be admitted only when present in the explicit
        # file ledger. A submodule transition removes the prior path binding.
        result = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v2",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V1),
                ("scratch/local.py", "def local():\n    return 1\n"),
            ),
        )
        assert result.head_advanced is True
        assert "vendor/lib/mod.py" in result.delta.deleted
        assert any(
            item.path == "vendor/lib/mod.py"
            and item.reason is InvalidationReason.PATH_DELETED
            for item in result.invalidations
        )
        paths = {
            item["path"]
            for item in indexer.ast_index.list_files(result.snapshot_id)
        }
        assert "scratch/local.py" in paths
        assert "vendor/lib/mod.py" not in paths


def test_dependent_facts_invalidated_on_source_parser_policy_drift(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as indexer:
        base = indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v1",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V1),
                ("src/consumer.py", PYTHON_CONSUMER),
            ),
            policy_id="policy:v1",
            parser_id="python-ast@schema-test-1",
        )
        symbol_fact = indexer.register_dependent_fact(
            worktree_id="worktree:wt-1",
            fact_kind=FactKind.SYMBOL,
            subject_path="src/service.py",
            subject_id="Service.dispatch",
            bound_snapshot_id=base.snapshot_id,
            bound_parser_id="python-ast@schema-test-1",
            bound_policy_id="policy:v1",
        )
        impact_fact = indexer.register_dependent_fact(
            worktree_id="worktree:wt-1",
            fact_kind=FactKind.IMPACT,
            subject_path="src/consumer.py",
            subject_id="impact:consume",
            bound_snapshot_id=base.snapshot_id,
            bound_parser_id="python-ast@schema-test-1",
            bound_policy_id="policy:v1",
        )
        cache_fact = indexer.register_dependent_fact(
            worktree_id="worktree:wt-1",
            fact_kind=FactKind.CACHE,
            subject_path="src/service.py",
            subject_id="cache:service",
            bound_snapshot_id=base.snapshot_id,
            bound_parser_id="python-ast@schema-test-1",
            bound_policy_id="policy:v1",
        )
        proof_fact = indexer.register_dependent_fact(
            worktree_id="worktree:wt-1",
            fact_kind=FactKind.PROOF,
            subject_path="src/service.py",
            subject_id="proof:dispatch",
            bound_snapshot_id=base.snapshot_id,
            bound_parser_id="python-ast@schema-test-1",
            bound_policy_id="policy:v1",
        )
        assert all(
            item.currency is FactCurrency.CURRENT
            for item in (
                symbol_fact,
                impact_fact,
                cache_fact,
                proof_fact,
            )
        )

        # Source drift on service invalidates path-bound facts only.
        source_scan = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v2",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V2),
                ("src/consumer.py", PYTHON_CONSUMER),
            ),
            policy_id="policy:v1",
            parser_id="python-ast@schema-test-1",
        )
        assert source_scan.head_advanced is True
        by_id = {
            item.fact_id: item
            for item in indexer.list_dependent_facts("worktree:wt-1")
        }
        assert by_id[symbol_fact.fact_id].currency is FactCurrency.INVALIDATED
        assert by_id[cache_fact.fact_id].currency is FactCurrency.INVALIDATED
        assert by_id[proof_fact.fact_id].currency is FactCurrency.INVALIDATED
        # Unchanged consumer impact remains current and rebinds to new snapshot.
        assert by_id[impact_fact.fact_id].currency is FactCurrency.CURRENT
        assert (
            by_id[impact_fact.fact_id].bound_snapshot_id
            == source_scan.snapshot_id
        )

        # Re-register current facts, then force parser drift.
        for kind, path, subject in (
            (FactKind.SYMBOL, "src/service.py", "Service.dispatch"),
            (FactKind.IMPACT, "src/consumer.py", "impact:consume"),
            (FactKind.CACHE, "src/service.py", "cache:service"),
            (FactKind.PROOF, "src/service.py", "proof:dispatch"),
        ):
            indexer.register_dependent_fact(
                worktree_id="worktree:wt-1",
                fact_kind=kind,
                subject_path=path,
                subject_id=subject,
                bound_snapshot_id=source_scan.snapshot_id,
                bound_parser_id="python-ast@schema-test-1",
                bound_policy_id="policy:v1",
            )

        parser_scan = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v3",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V2),
                ("src/consumer.py", PYTHON_CONSUMER),
            ),
            policy_id="policy:v1",
            parser_id="python-ast@schema-test-2-drift",
        )
        assert parser_scan.head_advanced is True
        assert any(
            item.reason is InvalidationReason.PARSER_DRIFT
            for item in parser_scan.invalidations
        )
        current_after_parser = indexer.list_dependent_facts(
            "worktree:wt-1", currency=FactCurrency.CURRENT
        )
        assert current_after_parser == ()
        invalidated = indexer.list_dependent_facts(
            "worktree:wt-1", currency=FactCurrency.INVALIDATED
        )
        assert len(invalidated) >= 4
        assert all(
            item.invalidated_by
            in {
                InvalidationReason.PARSER_DRIFT.value,
                InvalidationReason.SOURCE_CHANGED.value,
            }
            for item in invalidated
        )

        # Policy drift also clears currency of newly registered facts.
        for kind, path, subject in (
            (FactKind.SYMBOL, "src/service.py", "Service.dispatch"),
            (FactKind.PROOF, "src/service.py", "proof:dispatch"),
        ):
            indexer.register_dependent_fact(
                worktree_id="worktree:wt-1",
                fact_kind=kind,
                subject_path=path,
                subject_id=subject + ":policy",
                bound_snapshot_id=parser_scan.snapshot_id,
                bound_parser_id="python-ast@schema-test-2-drift",
                bound_policy_id="policy:v1",
            )
        policy_scan = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:v4",
            files=_files(
                ("src/service.py", PYTHON_SERVICE_V2),
                ("src/consumer.py", PYTHON_CONSUMER),
            ),
            policy_id="policy:v2-drift",
            parser_id="python-ast@schema-test-2-drift",
        )
        assert policy_scan.head_advanced is True
        assert any(
            item.reason is InvalidationReason.POLICY_DRIFT
            for item in policy_scan.invalidations
        )
        assert (
            indexer.list_dependent_facts(
                "worktree:wt-1", currency=FactCurrency.CURRENT
            )
            == ()
        )


def test_incremental_noop_when_identities_unchanged(tmp_path: Path) -> None:
    files = _files(
        ("src/service.py", PYTHON_SERVICE_V1),
        ("src/consumer.py", PYTHON_CONSUMER),
    )
    with _open(tmp_path) as indexer:
        first = indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:stable",
            files=files,
        )
        second = indexer.incremental_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:stable",
            files=files,
        )
        assert second.head_advanced is True
        assert second.delta.added == ()
        assert second.delta.changed == ()
        assert second.delta.deleted == ()
        assert set(second.delta.unchanged) == {
            "src/service.py",
            "src/consumer.py",
        }
        # Parse cache should satisfy both units on the second pass.
        assert second.coverage.reused_count >= 1
        equivalence = indexer.snapshot_equivalence(
            first.snapshot_id, second.snapshot_id
        )
        assert equivalence["equal"] is True


def test_scan_result_dict_is_body_free(tmp_path: Path) -> None:
    with _open(tmp_path) as indexer:
        result = indexer.full_scan(
            worktree_id="worktree:wt-1",
            repository_id="repo:demo",
            tree_id="tree:body",
            files=_files(("src/service.py", PYTHON_SERVICE_V1)),
        )
        payload = result.to_dict()
        encoded = str(payload)
        assert "class Service" not in encoded
        assert "def dispatch" not in encoded
        assert payload["head_advanced"] is True
        assert payload["coverage"]["complete"] is True
