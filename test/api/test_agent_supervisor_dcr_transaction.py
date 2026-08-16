"""DCR-072 isolated transaction-controller tests; all roots are temporary."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.root_ownership import RootBinding
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.transaction import (
    DCR072_ISOLATED_MARKER_SCHEMA,
    ISOLATED_WORKTREE_MARKER,
    Dcr003WorktreeBinding,
    Dcr070AdmissionBinding,
    Dcr071FencedWrite,
    Dcr071OperatorPreview,
    IsolatedTransactionController,
    TransactionDenied,
    TransactionDisposition,
    TransactionRequest,
    TransactionState,
    canonical_transaction_journal_bytes,
    isolated_baseline_digest,
)


def _request(root: Path, *, cancelled: bool = False) -> TransactionRequest:
    before = (root / "governed.txt").read_bytes()
    baseline = isolated_baseline_digest(root)
    marker = {
        "baseline_digest": baseline,
        "owner": "swissknife",
        "root_realpath": str(root.resolve()),
        "schema": DCR072_ISOLATED_MARKER_SCHEMA,
    }
    (root / ISOLATED_WORKTREE_MARKER).write_text(
        json.dumps(marker, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    binding = Dcr003WorktreeBinding(
        owner="swissknife",
        root=RootBinding(
            root_id="swissknife",
            realpath=str(root.resolve()),
            head="a" * 40,
            tree="b" * 40,
            overlay_digest=baseline,
            dirty=False,
        ),
        clean_baseline_digest=baseline,
    )
    write = Dcr071FencedWrite(
        relative_path="governed.txt",
        before_digest="sha256:" + hashlib.sha256(before).hexdigest(),
        after_bytes=b"after\n",
        after_digest="sha256:" + hashlib.sha256(b"after\n").hexdigest(),
        inverse_bytes=before,
        inverse_digest="sha256:" + hashlib.sha256(before).hexdigest(),
    )
    return TransactionRequest(
        transaction_id="txn-001",
        lease_id="lease-001",
        fence_id="fence-001",
        dcr003=binding,
        dcr070=Dcr070AdmissionBinding("cid:unintegrated-dcr070"),
        dcr071=Dcr071OperatorPreview("cid:dcr071-preview", "transport.normalize", (write,)),
        cancelled=cancelled,
    )


def _root(tmp_path: Path) -> Path:
    root = tmp_path / "isolated-owner-worktree"
    root.mkdir()
    (root / "governed.txt").write_bytes(b"before\n")
    return root


def test_unintegrated_admission_is_validation_pending_and_never_promotes(tmp_path: Path) -> None:
    root = _root(tmp_path)
    request = _request(root)
    original = (root / "governed.txt").read_bytes()

    journal = IsolatedTransactionController().run(request, isolated_root=root)

    assert journal.state is TransactionState.INTEGRATION_PENDING
    assert journal.disposition is TransactionDisposition.INTEGRATION_PENDING
    assert journal.execution_authorized is journal.completion_authorized is False
    assert journal.rollback_verified is True
    assert all(receipt.promoted is False for receipt in journal.writes)
    assert (root / "governed.txt").read_bytes() == original
    assert canonical_transaction_journal_bytes(journal) == canonical_transaction_journal_bytes(journal)
    assert journal.journal_cid


@pytest.mark.parametrize("fault", ("stale", "wrong_owner", "symlink"))
def test_stale_or_unsafe_isolated_roots_are_rejected_without_residue(
    tmp_path: Path, fault: str
) -> None:
    root = _root(tmp_path)
    request = _request(root)
    original = (root / "governed.txt").read_bytes()
    if fault == "stale":
        (root / "governed.txt").write_bytes(b"changed outside fence\n")
    elif fault == "wrong_owner":
        marker_path = root / ISOLATED_WORKTREE_MARKER
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        marker["owner"] = "other-owner"
        marker_path.write_text(json.dumps(marker), encoding="utf-8")
    else:
        (root / "governed.txt").unlink()
        (root / "governed.txt").symlink_to(tmp_path / "outside.txt")

    journal = IsolatedTransactionController().run(request, isolated_root=root)

    assert journal.state is TransactionState.REJECTED
    assert journal.disposition is TransactionDisposition.REJECTED
    assert journal.execution_authorized is journal.completion_authorized is False
    if fault == "stale":
        assert (root / "governed.txt").read_bytes() != request.dcr071.writes[0].after_bytes
    elif fault == "wrong_owner":
        assert (root / "governed.txt").read_bytes() == original
    else:
        assert (root / "governed.txt").is_symlink()


def test_cancellation_and_fake_admission_cannot_authorize_mutation(tmp_path: Path) -> None:
    root = _root(tmp_path)
    request = _request(root, cancelled=True)
    journal = IsolatedTransactionController().run(request, isolated_root=root)

    assert journal.state is TransactionState.CANCELLED
    assert journal.disposition is TransactionDisposition.CANCELLED
    assert journal.rollback_verified is True
    assert (root / "governed.txt").read_bytes() == b"before\n"
    with pytest.raises(TransactionDenied, match="always integration pending"):
        Dcr070AdmissionBinding("cid:forged", integration_pending=False)


def test_duplicate_paths_and_dirty_dcr003_binding_are_closed_at_input_boundary(tmp_path: Path) -> None:
    root = _root(tmp_path)
    request = _request(root)
    write = request.dcr071.writes[0]
    with pytest.raises(TransactionDenied, match="duplicate"):
        Dcr071OperatorPreview("cid:duplicate", "operator", (write, write))
    with pytest.raises(TransactionDenied, match="must be clean"):
        Dcr003WorktreeBinding(
            owner="swissknife",
            root=RootBinding(
                root_id="swissknife",
                realpath=str(root),
                head="a" * 40,
                tree="b" * 40,
                overlay_digest=request.dcr003.clean_baseline_digest,
                dirty=True,
            ),
            clean_baseline_digest=request.dcr003.clean_baseline_digest,
        )
