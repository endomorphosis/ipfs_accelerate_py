"""Tests for MutationLedger@1 (DQP-022).

Evidence subset: no-op edit, partial write, rename, multi-file SCC,
formatting-only change, parse failure, rollback, stable structural identity.

Acceptance:

* Every admitted byte change has one lineage or is rejected/quarantined
* Line-number churn alone does not forge a distinct semantic mutation
* Stale fence or mismatched before snapshot cannot record an accepted mutation
* Rollback restoration is independently verified
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.mutation_ledger import (
    AST_MUTATION_INTERFACE,
    AUTHORITY_CLASS,
    MUTATION_FILE_INTERFACE,
    MUTATION_LEDGER_INTERFACE,
    MUTATION_SET_INTERFACE,
    FileChangeKind,
    MutationContext,
    MutationDisposition,
    MutationFileSpec,
    MutationLedger,
    MutationStatus,
    ParseOutcome,
    RollbackStatus,
    content_digest_of,
    duckdb_available,
    language_for_path,
    open_mutation_ledger,
    semantic_mutation_identity,
    structural_identity_of,
)


pytestmark = pytest.mark.skipif(
    not duckdb_available(),
    reason="DuckDB is required for MutationLedger hermetic tests",
)


PYTHON_V1 = """\
class Service:
    def dispatch(self, request):
        return request
"""

PYTHON_V2 = """\
class Service:
    def dispatch(self, request):
        self.status = "running"
        return request
"""

# Same AST as PYTHON_V1 but with extra blank lines (line-number churn only).
PYTHON_V1_LINE_CHURN = """\
class Service:


    def dispatch(self, request):

        return request
"""

# Formatting-only whitespace change that preserves AST dump.
PYTHON_V1_FORMATTED = """\
class Service:
    def dispatch(self, request):
        return request  # trailing spaces stripped structurally via AST
"""

PYTHON_HELPER = """\
def helper():
    return 42
"""

PYTHON_HELPER_RENAMED = PYTHON_HELPER

PYTHON_BROKEN = """\
def broken(
    return None
"""

PYTHON_CONSUMER = """\
from src.service import Service

def consume(request):
    service = Service()
    return service.dispatch(request)
"""

PYTHON_CONSUMER_V2 = """\
from src.service import Service
from src.helper import helper

def consume(request):
    service = Service()
    return service.dispatch(request) + helper()
"""


def _open(tmp_path: Path) -> MutationLedger:
    return open_mutation_ledger(tmp_path / "mutation_ledger.duckdb")


def _fence(ledger: MutationLedger, *, worktree_id: str = "worktree:wt-1", **kwargs):
    return ledger.register_fence(
        worktree_id=worktree_id,
        token="fence-token-alpha",
        before_snapshot_id=kwargs.pop("before_snapshot_id", "snapshot:before-1"),
        before_tree_id=kwargs.pop("before_tree_id", "tree:before-1"),
        lease_id=kwargs.pop("lease_id", "lease:1"),
        session_id=kwargs.pop("session_id", "session:1"),
        **kwargs,
    )


def _context(fence, **overrides) -> MutationContext:
    base = {
        "task_id": "task:DQP-022",
        "attempt_id": "attempt:1",
        "plan_id": "plan:step-3",
        "operator_id": "operator:daemon",
        "provider_id": "provider:test",
        "daemon_id": "daemon:impl-1",
        "session_id": fence.session_id or "session:1",
        "worktree_id": fence.worktree_id,
        "lease_id": fence.lease_id or "lease:1",
        "fence_id": fence.fence_id,
        "before_snapshot_id": fence.before_snapshot_id or "snapshot:before-1",
        "after_snapshot_id": "snapshot:after-1",
        "before_tree_id": fence.before_tree_id or "tree:before-1",
        "after_tree_id": "tree:after-1",
        "repository_id": "repo:demo",
        "declared_effects": {"symbols": ["Service.dispatch"]},
        "validation_outcome": "pending",
        "proof_outcome": "pending",
        "merge_outcome": "pending",
    }
    base.update(overrides)
    return MutationContext(**base)


def test_interface_identities() -> None:
    assert MUTATION_LEDGER_INTERFACE == "MutationLedger@1"
    assert MUTATION_SET_INTERFACE == "MutationSet@1"
    assert MUTATION_FILE_INTERFACE == "MutationFile@1"
    assert AST_MUTATION_INTERFACE == "ASTMutation@1"
    assert MutationLedger.INTERFACE == MUTATION_LEDGER_INTERFACE
    assert AUTHORITY_CLASS == "derived_evidence"
    assert language_for_path("src/app.py") == "python"


def test_cold_import_and_construction_have_no_side_effects() -> None:
    store = MutationLedger("/tmp/should-not-exist-until-open.duckdb")
    assert store.is_open is False


def test_admitted_byte_change_has_exactly_one_lineage(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                )
            ],
        )
        assert result.admitted is True
        assert result.mutation.status is MutationStatus.ACCEPTED
        assert result.mutation.disposition is MutationDisposition.ACCEPTED
        assert result.mutation.interface == MUTATION_SET_INTERFACE
        assert result.mutation.to_dict()["authority"] == AUTHORITY_CLASS

        # Exactly one lineage for the one byte-changing path.
        assert len(result.lineages) == 1
        lineage = result.lineages[0]
        assert lineage.path == "src/service.py"
        assert lineage.byte_changed is True
        assert lineage.semantic_changed is True
        assert lineage.ast_mutation_id
        assert lineage.disposition is MutationDisposition.ACCEPTED

        files = ledger.list_mutation_files(result.mutation.mutation_id)
        assert len(files) == 1
        assert files[0].lineage_id == lineage.lineage_id
        assert files[0].interface == MUTATION_FILE_INTERFACE
        assert files[0].change_kind is FileChangeKind.MODIFIED

        ast_muts = ledger.list_ast_mutations(result.mutation.mutation_id)
        assert len(ast_muts) == 1
        assert ast_muts[0].interface == AST_MUTATION_INTERFACE
        assert ast_muts[0].semantic_changed is True
        assert "Service.dispatch" in ast_muts[0].symbols_changed or any(
            op.get("op") == "replace_symbol"
            for op in ast_muts[0].edit_script.get("ops", [])
        )
        hunks = ledger.list_hunks(result.mutation.mutation_id)
        assert hunks
        assert all(h.path == "src/service.py" for h in hunks)

        # Re-load from store.
        loaded = ledger.get_mutation(result.mutation.mutation_id)
        assert loaded is not None
        assert loaded.semantic_mutation_id == result.mutation.semantic_mutation_id
        assert loaded.fence_id == fence.fence_id
        meta = ledger.metadata()
        assert meta["interface"] == MUTATION_LEDGER_INTERFACE


def test_line_number_churn_does_not_forge_distinct_semantic_mutation(
    tmp_path: Path,
) -> None:
    """Same structural AST under different line numbers shares semantic id."""

    with _open(tmp_path) as ledger:
        fence_a = _fence(ledger, worktree_id="worktree:a")
        # Semantic edit: V1 -> V2
        first = ledger.record_mutation(
            _context(
                fence_a,
                task_id="task:sem-1",
                attempt_id="attempt:a",
                after_snapshot_id="snapshot:a1",
            ),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                )
            ],
        )
        assert first.admitted is True

        fence_b = _fence(ledger, worktree_id="worktree:b")
        # Same structural before/after, but before side has line churn only.
        second = ledger.record_mutation(
            _context(
                fence_b,
                task_id="task:sem-2",
                attempt_id="attempt:b",
                worktree_id="worktree:b",
                before_snapshot_id=fence_b.before_snapshot_id,
                after_snapshot_id="snapshot:b1",
            ),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1_LINE_CHURN,
                    after_content=PYTHON_V2,
                )
            ],
        )
        assert second.admitted is True

        # Structural identity of PYTHON_V1 and line-churn variant match.
        s1, p1, _ = structural_identity_of(PYTHON_V1, language="python")
        s2, p2, _ = structural_identity_of(
            PYTHON_V1_LINE_CHURN, language="python"
        )
        assert p1 is ParseOutcome.SUCCEEDED
        assert p2 is ParseOutcome.SUCCEEDED
        assert s1 == s2

        assert (
            first.mutation.semantic_mutation_id
            == second.mutation.semantic_mutation_id
        )
        assert first.files[0].before_structural_id == second.files[0].before_structural_id
        assert first.files[0].after_structural_id == second.files[0].after_structural_id

        # Pure line-churn as the entire mutation is formatting-only / no semantic.
        fence_c = _fence(ledger, worktree_id="worktree:c")
        churn_only = ledger.record_mutation(
            _context(
                fence_c,
                task_id="task:churn",
                worktree_id="worktree:c",
                before_snapshot_id=fence_c.before_snapshot_id,
            ),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V1_LINE_CHURN,
                )
            ],
        )
        assert churn_only.admitted is True
        assert churn_only.mutation.disposition is MutationDisposition.FORMATTING_ONLY
        assert churn_only.files[0].formatting_only is True
        assert churn_only.files[0].semantic_changed is False
        # Lineage still present for the byte change.
        assert len(churn_only.lineages) == 1


def test_stale_fence_cannot_record_accepted_mutation(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        # Supersede by registering a new generation.
        new_fence = _fence(ledger, generation=None)
        assert new_fence.generation > fence.generation
        stale = ledger.get_fence(fence.fence_id)
        assert stale is not None
        assert stale.status.value == "superseded"

        result = ledger.record_mutation(
            _context(fence),  # still references the stale fence
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                )
            ],
        )
        assert result.admitted is False
        assert result.rejected is True
        assert result.mutation.status is MutationStatus.REJECTED
        assert result.mutation.disposition is MutationDisposition.STALE_FENCE
        # Must not appear as an accepted mutation.
        accepted = ledger.list_mutations(status=MutationStatus.ACCEPTED)
        assert all(m.mutation_id != result.mutation.mutation_id for m in accepted)
        quarantine = ledger.list_quarantine(worktree_id=fence.worktree_id)
        assert any(q["mutation_id"] == result.mutation.mutation_id for q in quarantine)


def test_mismatched_before_snapshot_cannot_record_accepted_mutation(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger, before_snapshot_id="snapshot:expected")
        result = ledger.record_mutation(
            _context(fence, before_snapshot_id="snapshot:wrong"),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                )
            ],
        )
        assert result.admitted is False
        assert result.rejected is True
        assert result.mutation.status is MutationStatus.REJECTED
        assert (
            result.mutation.disposition is MutationDisposition.SNAPSHOT_MISMATCH
        )


def test_rollback_restoration_independently_verified(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                ),
                MutationFileSpec(
                    path="src/helper.py",
                    before_content=None,
                    after_content=PYTHON_HELPER,
                ),
            ],
        )
        assert result.admitted is True
        mid = result.mutation.mutation_id

        # Correct restoration: service back to V1, helper removed.
        verified = ledger.record_rollback(
            mutation_id=mid,
            restored_files={
                "src/service.py": PYTHON_V1,
                "src/helper.py": None,
            },
        )
        assert verified.verified is True
        assert verified.status is RollbackStatus.VERIFIED
        assert verified.mismatches == {}
        loaded = ledger.get_mutation(mid)
        assert loaded is not None
        assert loaded.status is MutationStatus.ROLLED_BACK
        assert loaded.rollback_outcome == RollbackStatus.VERIFIED.value

        receipt = ledger.get_rollback(mid)
        assert receipt is not None
        assert receipt.verified is True
        assert receipt.expected_digests["src/service.py"] == content_digest_of(
            PYTHON_V1
        )
        assert receipt.expected_digests["src/helper.py"] == ""

        # A second mutation to show failed independent verification.
        fence2 = _fence(ledger)
        second = ledger.record_mutation(
            _context(
                fence2,
                task_id="task:rollback-fail",
                attempt_id="attempt:2",
                before_snapshot_id=fence2.before_snapshot_id,
            ),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                )
            ],
        )
        failed = ledger.record_rollback(
            mutation_id=second.mutation.mutation_id,
            # Still at after state — not restored.
            restored_files={"src/service.py": PYTHON_V2},
        )
        assert failed.verified is False
        assert failed.status is RollbackStatus.FAILED
        assert "src/service.py" in failed.mismatches
        still = ledger.get_mutation(second.mutation.mutation_id)
        assert still is not None
        # Failed rollback must not mark the mutation rolled_back.
        assert still.status is MutationStatus.ACCEPTED


def test_no_op_edit(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V1,
                )
            ],
        )
        assert result.admitted is False  # no_op is not "accepted" admission
        assert result.mutation.status is MutationStatus.NO_OP
        assert result.mutation.disposition is MutationDisposition.NO_OP
        assert result.lineages == ()
        assert result.files[0].change_kind is FileChangeKind.NO_OP
        assert (
            result.files[0].before_content_digest
            == result.files[0].after_content_digest
        )


def test_partial_write_is_quarantined(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                ),
                MutationFileSpec(
                    path="src/helper.py",
                    before_content=None,
                    after_content=PYTHON_HELPER,
                    partial=True,
                ),
            ],
        )
        assert result.admitted is False
        assert result.quarantined is True
        assert result.mutation.status is MutationStatus.QUARANTINED
        assert result.mutation.disposition is MutationDisposition.PARTIAL_WRITE
        quarantine = ledger.list_quarantine(worktree_id=fence.worktree_id)
        assert any(
            q["mutation_id"] == result.mutation.mutation_id for q in quarantine
        )


def test_rename_records_prior_path_and_lineage(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/new_name.py",
                    prior_path="src/old_name.py",
                    before_content=PYTHON_HELPER,
                    after_content=PYTHON_HELPER_RENAMED,
                )
            ],
        )
        assert result.admitted is True
        mf = result.files[0]
        assert mf.change_kind is FileChangeKind.RENAMED
        assert mf.prior_path == "src/old_name.py"
        assert mf.path == "src/new_name.py"
        # Same bytes => not semantic, but rename is still a byte-path change
        # when digests match the rename still has before==after digests.
        # For pure rename with identical content digests, byte_changed is False
        # so status may be NO_OP. Force a content-preserving rename by ensuring
        # path change is tracked: when digests equal, ledger treats as no_op
        # unless prior_path forces rename kind with lineage only on byte change.
        assert mf.change_kind is FileChangeKind.RENAMED

        # Rename with content change.
        fence2 = _fence(ledger)
        changed = ledger.record_mutation(
            _context(
                fence2,
                task_id="task:rename-edit",
                before_snapshot_id=fence2.before_snapshot_id,
            ),
            [
                MutationFileSpec(
                    path="pkg/service.py",
                    prior_path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                )
            ],
        )
        assert changed.admitted is True
        assert changed.files[0].change_kind is FileChangeKind.RENAMED
        assert changed.files[0].semantic_changed is True
        assert len(changed.lineages) == 1
        assert changed.files[0].prior_path == "src/service.py"


def test_multi_file_scc_mutation_binds_all_lineages(tmp_path: Path) -> None:
    """Interdependent multi-file change records one set with per-file lineage."""

    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(
                fence,
                declared_effects={
                    "scc": ["Service.dispatch", "consume", "helper"],
                    "files": [
                        "src/service.py",
                        "src/consumer.py",
                        "src/helper.py",
                    ],
                },
            ),
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                ),
                MutationFileSpec(
                    path="src/consumer.py",
                    before_content=PYTHON_CONSUMER,
                    after_content=PYTHON_CONSUMER_V2,
                ),
                MutationFileSpec(
                    path="src/helper.py",
                    before_content=None,
                    after_content=PYTHON_HELPER,
                ),
            ],
        )
        assert result.admitted is True
        assert result.mutation.file_count == 3
        assert result.mutation.lineage_count == 3
        assert len(result.lineages) == 3
        paths = {lin.path for lin in result.lineages}
        assert paths == {
            "src/service.py",
            "src/consumer.py",
            "src/helper.py",
        }
        # Every byte-changing file has exactly one lineage.
        lineage_by_path = {lin.path: lin for lin in result.lineages}
        for mf in result.files:
            assert mf.lineage_id == lineage_by_path[mf.path].lineage_id
        # Declared effects and binding identities are retained.
        body = result.mutation.to_dict()
        assert body["declared_effects"]["scc"]
        assert body["task_id"] == "task:DQP-022"
        assert body["provider_id"] == "provider:test"
        assert body["daemon_id"] == "daemon:impl-1"


def test_formatting_only_change_has_lineage_without_semantic_shift(
    tmp_path: Path,
) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        # Ensure digests differ while structural id matches.
        before = "def f():\n    return 1\n"
        after = "def f():\n    return 1\n\n\n"
        s_before, _, _ = structural_identity_of(before, language="python")
        s_after, _, _ = structural_identity_of(after, language="python")
        assert s_before == s_after
        assert content_digest_of(before) != content_digest_of(after)

        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/util.py",
                    before_content=before,
                    after_content=after,
                )
            ],
        )
        assert result.admitted is True
        assert result.mutation.disposition is MutationDisposition.FORMATTING_ONLY
        assert result.files[0].formatting_only is True
        assert result.files[0].semantic_changed is False
        assert len(result.lineages) == 1
        assert result.lineages[0].semantic_changed is False


def test_parse_failure_is_quarantined_not_accepted(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/broken.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_BROKEN,
                )
            ],
        )
        assert result.admitted is False
        assert result.quarantined is True
        assert result.mutation.status is MutationStatus.QUARANTINED
        assert result.mutation.disposition is MutationDisposition.PARSE_FAILED
        assert result.files[0].parse_status is ParseOutcome.FAILED
        # No accepted lineage for parse-failed byte change.
        assert result.lineages == ()
        quarantine = ledger.list_quarantine(worktree_id=fence.worktree_id)
        assert any(
            "parse" in str(q.get("reason") or "").lower()
            or q["mutation_id"] == result.mutation.mutation_id
            for q in quarantine
        )


def test_stable_structural_identity_helpers() -> None:
    a, status_a, rec_a = structural_identity_of(PYTHON_V1, language="python")
    b, status_b, rec_b = structural_identity_of(
        PYTHON_V1_LINE_CHURN, language="python"
    )
    c, status_c, rec_c = structural_identity_of(PYTHON_V2, language="python")
    assert status_a is ParseOutcome.SUCCEEDED
    assert status_b is ParseOutcome.SUCCEEDED
    assert status_c is ParseOutcome.SUCCEEDED
    assert a == b
    assert a != c
    assert rec_a is not None and rec_c is not None
    assert rec_a.symbol_hashes.get("Service.dispatch") != rec_c.symbol_hashes.get(
        "Service.dispatch"
    )

    # semantic_mutation_identity ignores pure non-semantic members when empty.
    id1 = semantic_mutation_identity(
        [
            {
                "path": "src/service.py",
                "change_kind": FileChangeKind.MODIFIED.value,
                "before_structural_id": a,
                "after_structural_id": c,
                "semantic_changed": True,
            }
        ]
    )
    id2 = semantic_mutation_identity(
        [
            {
                "path": "src/service.py",
                "change_kind": FileChangeKind.MODIFIED.value,
                "before_structural_id": b,  # same structural as a
                "after_structural_id": c,
                "semantic_changed": True,
            }
        ]
    )
    assert id1 == id2


def test_unknown_fence_is_rejected(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        ctx = MutationContext(
            task_id="task:x",
            worktree_id="worktree:wt-1",
            fence_id="mutation-fence:sha256:" + "0" * 64,
            before_snapshot_id="snapshot:before-1",
        )
        result = ledger.record_mutation(
            ctx,
            [
                MutationFileSpec(
                    path="src/service.py",
                    before_content=PYTHON_V1,
                    after_content=PYTHON_V2,
                )
            ],
        )
        assert result.admitted is False
        assert result.mutation.disposition is MutationDisposition.STALE_FENCE


def test_added_and_deleted_files(tmp_path: Path) -> None:
    with _open(tmp_path) as ledger:
        fence = _fence(ledger)
        result = ledger.record_mutation(
            _context(fence),
            [
                MutationFileSpec(
                    path="src/new.py",
                    before_content=None,
                    after_content=PYTHON_HELPER,
                ),
                MutationFileSpec(
                    path="src/gone.py",
                    before_content=PYTHON_HELPER,
                    after_content=None,
                ),
            ],
        )
        assert result.admitted is True
        kinds = {mf.path: mf.change_kind for mf in result.files}
        assert kinds["src/new.py"] is FileChangeKind.ADDED
        assert kinds["src/gone.py"] is FileChangeKind.DELETED
        assert len(result.lineages) == 2
        ast_by_path = {am.path: am for am in result.ast_mutations}
        assert any(
            op.get("op") == "add_file"
            for op in ast_by_path["src/new.py"].edit_script.get("ops", [])
        )
        assert any(
            op.get("op") == "delete_file"
            for op in ast_by_path["src/gone.py"].edit_script.get("ops", [])
        )
