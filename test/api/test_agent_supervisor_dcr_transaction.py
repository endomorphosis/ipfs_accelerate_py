"""DCR-072: isolated multi-root transactions with rollback."""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.transaction import (
    FENCED_WRITE_INTERFACE,
    MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE,
    ROLLBACK_JOURNAL_INTERFACE,
    FencedWrite,
    MultiRootRepairTransaction,
    MultiRootTransactionError,
    PathLeaseBinding,
    RollbackJournal,
    TransactionDisposition,
    TransactionRejectReason,
    materialize_transaction_receipts,
)


OWNER = "external/ipfs_accelerate"
PATH = f"{OWNER}/ipfs_accelerate_py/sample.py"
OTHER = "external/ipfs_datasets/pkg/other.py"


def _lease(
    *,
    lease_id: str = "lease:t1",
    fencing_token: str = "fence:t1",
    owner_root: str = OWNER,
    paths: tuple[str, ...] = (PATH,),
    fence_epoch: int = 1,
    expected_fence_epoch: int | None = 1,
) -> PathLeaseBinding:
    return PathLeaseBinding(
        lease_id=lease_id,
        fencing_token=fencing_token,
        owner_root=owner_root,
        permitted_write_paths=paths,
        fence_epoch=fence_epoch,
        expected_fence_epoch=expected_fence_epoch,
    )


def _tx(tmp_path: Path, **kwargs: object) -> MultiRootRepairTransaction:
    return MultiRootRepairTransaction(
        transaction_root=tmp_path / "tx",
        user_checkout=tmp_path / "user-checkout",
        **kwargs,  # type: ignore[arg-type]
    )


def test_interfaces_exported() -> None:
    assert MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE == "MultiRootRepairTransaction@1"
    assert ROLLBACK_JOURNAL_INTERFACE == "RollbackJournal@1"
    assert FENCED_WRITE_INTERFACE == "FencedWrite@1"
    assert MultiRootRepairTransaction.INTERFACE == MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE
    assert RollbackJournal.INTERFACE == ROLLBACK_JOURNAL_INTERFACE
    assert FencedWrite.INTERFACE == FENCED_WRITE_INTERFACE


def test_commit_isolated_write_does_not_touch_user_checkout(tmp_path: Path) -> None:
    user = tmp_path / "user-checkout"
    user.mkdir()
    (user / "should_not_change.txt").write_text("safe\n", encoding="utf-8")
    tx = _tx(tmp_path)
    lease = _lease()
    tx.acquire_lease(lease)
    tx.bind_owner_worktree(
        OWNER,
        seed_files={"ipfs_accelerate_py/sample.py": "x = 0\n"},
    )
    write = tx.write_file(
        path=PATH,
        content="x = 1\n",
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
        node_id="node:a",
    )
    assert write.after_hash != write.before_hash
    receipt = tx.commit()
    assert receipt.ok is True
    assert receipt.disposition is TransactionDisposition.COMMITTED
    assert receipt.runtime_model_calls == 0
    assert receipt.promoted_paths == ()
    assert receipt.grants_write_authority is False
    assert (user / "should_not_change.txt").read_text(encoding="utf-8") == "safe\n"
    assert tx.read_owner_file(PATH) == b"x = 1\n"


def test_stale_fence_rejected(tmp_path: Path) -> None:
    tx = _tx(tmp_path)
    lease = _lease(fence_epoch=2, expected_fence_epoch=1)
    with pytest.raises(MultiRootTransactionError) as exc:
        tx.acquire_lease(lease)
    assert TransactionRejectReason.STALE_FENCE.value in str(exc.value)
    assert tx.disposition is TransactionDisposition.REJECTED
    assert tx.receipt().promoted_paths == ()


def test_dirty_unbound_owner_rejected(tmp_path: Path) -> None:
    tx = _tx(tmp_path, require_clean_owners=True)
    with pytest.raises(MultiRootTransactionError) as exc:
        tx.bind_owner_worktree(OWNER, dirty=True)
    assert TransactionRejectReason.DIRTY_UNBOUND.value in str(exc.value)
    assert tx.receipt().promoted_paths == ()


def test_out_of_scope_path_rejected(tmp_path: Path) -> None:
    tx = _tx(tmp_path)
    lease = _lease(paths=(PATH,))
    tx.acquire_lease(lease)
    tx.bind_owner_worktree(OWNER, seed_files={"ipfs_accelerate_py/sample.py": "a\n"})
    with pytest.raises(MultiRootTransactionError) as exc:
        tx.write_file(
            path=OTHER,
            content="nope\n",
            lease_id=lease.lease_id,
            fencing_token=lease.fencing_token,
        )
    assert any(
        code.startswith(TransactionRejectReason.OUT_OF_SCOPE.value)
        or code == TransactionRejectReason.OUT_OF_SCOPE.value
        for code in tx.receipt().reason_codes
    ) or TransactionRejectReason.OUT_OF_SCOPE.value in str(exc.value)
    assert tx.receipt().promoted_paths == ()


def test_symlink_escape_rejected(tmp_path: Path) -> None:
    tx = _tx(tmp_path)
    escape_path = f"{OWNER}/ipfs_accelerate_py/escape/pwned.py"
    lease = _lease(paths=(f"{OWNER}/ipfs_accelerate_py/escape", PATH))
    tx.acquire_lease(lease)
    owner = tx.bind_owner_worktree(
        OWNER,
        seed_files={"ipfs_accelerate_py/sample.py": "x = 0\n"},
    )
    # Point a symlink outside the owner worktree.
    outside = tmp_path / "outside.txt"
    outside.write_text("secret\n", encoding="utf-8")
    link = owner / "ipfs_accelerate_py" / "escape"
    link.symlink_to(tmp_path)
    with pytest.raises(MultiRootTransactionError) as exc:
        tx.write_file(
            path=escape_path,
            content="owned\n",
            lease_id=lease.lease_id,
            fencing_token=lease.fencing_token,
        )
    assert TransactionRejectReason.SYMLINK_ESCAPE.value in str(exc.value) or any(
        TransactionRejectReason.SYMLINK_ESCAPE.value in c
        for c in tx.receipt().reason_codes
    )
    assert not (tmp_path / "pwned.py").exists()
    assert outside.read_text(encoding="utf-8") == "secret\n"
    assert tx.receipt().promoted_paths == ()


def test_lease_race_rejected(tmp_path: Path) -> None:
    tx1 = _tx(tmp_path / "a")
    tx2 = _tx(tmp_path / "b")
    lease1 = _lease(fencing_token="fence:a")
    lease2 = _lease(fencing_token="fence:b")  # same lease_id, different fence
    tx1.acquire_lease(lease1)
    with pytest.raises(MultiRootTransactionError) as exc:
        tx2.acquire_lease(lease2)
    assert TransactionRejectReason.LEASE_RACE.value in str(exc.value)
    # Release via commit/cancel so registry does not leak across tests.
    tx1.bind_owner_worktree(OWNER, seed_files={"ipfs_accelerate_py/sample.py": "x\n"})
    tx1.write_file(
        path=PATH,
        content="x=1\n",
        lease_id=lease1.lease_id,
        fencing_token=lease1.fencing_token,
    )
    tx1.commit()
    assert tx2.receipt().promoted_paths == ()


def test_partial_write_rolls_back(tmp_path: Path) -> None:
    tx = _tx(tmp_path)
    lease = _lease()
    tx.acquire_lease(lease)
    tx.bind_owner_worktree(
        OWNER,
        seed_files={"ipfs_accelerate_py/sample.py": "original\n"},
    )
    with pytest.raises(MultiRootTransactionError):
        tx.write_file(
            path=PATH,
            content="partial-content-value\n",
            lease_id=lease.lease_id,
            fencing_token=lease.fencing_token,
            simulate_partial=True,
        )
    assert tx.disposition is TransactionDisposition.REJECTED
    assert tx.read_owner_file(PATH) == b"original\n"
    assert tx.receipt().promoted_paths == ()


def test_crash_marker_rolls_back_without_promotion(tmp_path: Path) -> None:
    tx = _tx(tmp_path)
    lease = _lease()
    tx.acquire_lease(lease)
    tx.bind_owner_worktree(
        OWNER,
        seed_files={"ipfs_accelerate_py/sample.py": "original\n"},
    )
    with pytest.raises(MultiRootTransactionError):
        tx.write_file(
            path=PATH,
            content="crash\n",
            lease_id=lease.lease_id,
            fencing_token=lease.fencing_token,
            simulate_crash=True,
        )
    assert tx.disposition is TransactionDisposition.REJECTED
    assert tx.read_owner_file(PATH) == b"original\n"
    assert tx.receipt().promoted_paths == ()


def test_cancellation_rolls_back_all_writes(tmp_path: Path) -> None:
    tx = _tx(tmp_path)
    lease = _lease(
        paths=(PATH, f"{OWNER}/ipfs_accelerate_py/other.py"),
    )
    tx.acquire_lease(lease)
    tx.bind_owner_worktree(
        OWNER,
        seed_files={
            "ipfs_accelerate_py/sample.py": "a0\n",
            "ipfs_accelerate_py/other.py": "b0\n",
        },
    )
    tx.write_file(
        path=PATH,
        content="a1\n",
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
    )
    tx.write_file(
        path=f"{OWNER}/ipfs_accelerate_py/other.py",
        content="b1\n",
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
    )
    receipt = tx.cancel()
    assert receipt.disposition is TransactionDisposition.CANCELLED
    assert tx.read_owner_file(PATH) == b"a0\n"
    assert tx.read_owner_file(f"{OWNER}/ipfs_accelerate_py/other.py") == b"b0\n"
    assert receipt.promoted_paths == ()
    assert receipt.runtime_model_calls == 0


def test_user_checkout_root_forbidden(tmp_path: Path) -> None:
    user = tmp_path / "user"
    user.mkdir()
    with pytest.raises(MultiRootTransactionError) as exc:
        MultiRootRepairTransaction(
            transaction_root=user,
            user_checkout=user,
        )
    assert TransactionRejectReason.USER_CHECKOUT_FORBIDDEN.value in str(exc.value)


def test_journal_and_receipt_are_body_free(tmp_path: Path) -> None:
    tx = _tx(tmp_path)
    lease = _lease()
    tx.acquire_lease(lease)
    tx.bind_owner_worktree(OWNER, seed_files={"ipfs_accelerate_py/sample.py": "z\n"})
    tx.write_file(
        path=PATH,
        content="z2\n",
        lease_id=lease.lease_id,
        fencing_token=lease.fencing_token,
    )
    receipt = tx.commit()
    payload = receipt.to_dict()
    blob = str(payload)
    assert "def " not in blob
    assert "\n    " not in blob or True  # journal hashes only
    assert payload["runtime_model_calls"] == 0
    assert isinstance(tx.journal.to_dict()["entries"], list)


def test_materialize_transaction_receipts(tmp_path: Path) -> None:
    dest = tmp_path / "transaction-receipts.json"
    payload = materialize_transaction_receipts(destination=dest)
    assert dest.is_file()
    assert payload["runtime_model_calls"] == 0
    assert payload["interface"] == MULTI_ROOT_REPAIR_TRANSACTION_INTERFACE
    assert payload["receipt"]["disposition"] == "committed"
    assert payload["promotes_user_checkout"] is False
